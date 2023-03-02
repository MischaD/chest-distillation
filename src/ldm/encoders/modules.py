import random

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
from log import logger
import open_clip
from src.ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, layer="last", multi_label_finetuning=False, rali=None):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version, cache_dir="./clip/")
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

        self.multi_label_finetuning = multi_label_finetuning
        self.multi_label_tokenizer = None
        self.attn_mask = torch.fill(torch.zeros(max_length, max_length), -1 * torch.inf).fill_diagonal_(0)
        if rali is not None:
            self.is_rali = True
            self.rali_random = rali == "random"

    def set_multi_label_tokenizer(self, multi_label_tokenizer):
        self.multi_label_tokenizer = multi_label_tokenizer

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def compute_word_len(self, words):
        if not isinstance(words, list):
            words = list(words)
        assert isinstance(words[0], str)
        outs = open_clip.tokenize(words)
        lens = []
        for out in outs:
            lens.append(int(sum(out != 0) - 2))
        return lens

    def check_is_rali(self, text):
        if not isinstance(text, list):
            return False
        if not isinstance(text[0], list):
            return False
        if not isinstance(text[0][0], str):
            return False
        return True

    def forward(self, text):
        if self.is_rali:
            for i, text_ in enumerate(text[1]):
                if isinstance(text_, float) or text_ == "":
                    # happens with nan --> indecisive samples == "No Finding"
                    text[1][i] = "No Finding"

            relevant_labels = ["No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Lung Opacity", "Pleural Effusion",
             "Pneumonia", "Pneumothorax"]
            new_text = []
            impressions = text[0]
            labels_batch = text[1]
            locs = []
            for i in range(len(impressions)):
                labels = labels_batch[i].split("|")
                labels = [l for l in labels if l in relevant_labels]
                random.shuffle(labels)

                loc = 0 if not self.rali_random else random.randint(0, 5)
                locs.append(loc)
                impression = impressions[i].split(" ")
                delimiter = "" if loc == 0 else " "
                new_text.append(
                    " ".join(impression[:loc]) + delimiter + " ".join(labels) + " " + " ".join(impression[loc:])
                )
            text = new_text

        if not self.multi_label_finetuning:
            tokens = open_clip.tokenize(text)
        else:
            for i, text_ in enumerate(text):
                if isinstance(text_, float) or text_ == "":
                    # happens with nan --> indecisive samples == "No Finding"
                    text[i] = "No Finding"
            tokens = torch.stack([self.multi_label_tokenizer(text_.split("|")) for text_ in text])
        z = self.encode_with_transformer(tokens.to(self.device), is_rali=self.is_rali)
        return z

    def encode_with_transformer(self, text, is_rali=False):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        if not self.multi_label_finetuning or is_rali:
            x = x + self.model.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask[:])
        else:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_transformer_forward(x, attn_mask=self.attn_mask[:x.size()[0], :x.size()[0]].to(self.device))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class OpenClipDummyTokenizer:
    N_TOKENS = 49408
    SOS_TOKEN = 49406
    EOS_TOKEN = 49407
    MAPPING = [
        {"No Finding": 1},
        {"Atelectasis": 2},
        {"Cardiomegaly": 3},
        {"Consolidation": 4},
        {"Edema": 5},
        {"Lung Opacity": 6},
        {"Pleural Effusion": 7},
        {"Pneumonia": 8},
        {"Pneumothorax": 9},
    ]

    MAPPINGv14 = [
        {"No Finding": 1},
        {"Atelectasis": 2},
        {"Cardiomegaly": 3},
        {"Consolidation": 4},
        {"Edema": 5},
        {"Lung Opacity": 6},
        {"Pleural Effusion": 7},
        {"Pneumonia": 8},
        {"Pneumothorax": 9},
        {"Enlarged Cardiomediastinum": 10},
        {"Fracture": 11},
        {"Lung Lesion": 12},
        {"Pleural Other": 13},
        {"Support Devices": 14},
    ]

    def __init__(self, seed, append_invariance_tokens, single_healthy_class_token, rali=None):
        r = np.random.RandomState(seed)
        mapping_to_token = np.arange(1, self.N_TOKENS - 2)
        r.shuffle(mapping_to_token)
        self.mapping_to_token = mapping_to_token
        self.append_invariance_tokens = append_invariance_tokens
        self.single_healthy_class_token = single_healthy_class_token
        if rali is not None:
            self.is_rali = True
            self.rali_random = rali == "random"

    def __call__(self, label_list):
        "Doublecheck empty --> Should be the same as No Finding"
        tokens = []
        for i in range(len(self.MAPPING)):
            for k, v in self.MAPPING[i].items():
                if k in label_list:
                    tokens.append(v)
                else:
                    if self.single_healthy_class_token and k not in ["Support Devices", "No Finding"]:
                        tokens.append(0)
                    else:
                        tokens.append(v + len(self.MAPPING))

        tokens = [self.mapping_to_token[token-1] for token in tokens]

        if self.append_invariance_tokens:
            tokens = [self.SOS_TOKEN,] + tokens + [self.EOS_TOKEN,]
            tokens = tokens + [0,] * (77 - len(tokens))
        return torch.tensor(tokens)

    def get_attention_map_location(self, label_list, handle_not_found=None):
        locations = []

        for label in label_list:
            for i in range(len(self.MAPPING)):
                k, v = [*self.MAPPING[i].items()][0]
                if k == label:
                    location = v
                    if not self.append_invariance_tokens:
                        location -= 1
                    locations.append(location)

        if handle_not_found is not None and locations == []:
            locations = [handle_not_found,]
        return locations

    #def populate_tokens(self, dataset, cond_key):
        #words = set()
        #for x in dataset:
        #    cond = x[cond_key]
        #    if isinstance(cond, float):
        #        # nan - missing label
        #        continue
        #    for c in cond.split("|"):
        #        words.add(c)

        #ordered_words = list(words)
        #sorted(ordered_words)
