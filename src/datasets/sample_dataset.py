import random
from log import logger


DISEASES_TO_GENERATE = ["No Finding","Atelectasis","Cardiomegaly", "Consolidation", "Edema", "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax"]


def get_mscxr_synth_dataset(opt, dataset, label_key="finding_labels", finding_key="label_text"):
    n = opt.n_synth_samples_per_class

    synth_dataset = {}

    for i in range(len(dataset)):

        sample = dataset[i]
        label = sample[label_key]
        if str(label) not in DISEASES_TO_GENERATE:
            continue

        label = random.choice(label.split("|"))
        if synth_dataset.get(label) is None:
            synth_dataset[label] = []

        label_texts = sample[finding_key]
        for label_text in label_texts.split("|"):
            synth_dataset[label].append(label_text)

    for label in synth_dataset.keys():
        random.shuffle(synth_dataset[label])
        while len(synth_dataset[label]) < n:
            synth_dataset[label] = synth_dataset[label] + synth_dataset[label]
        synth_dataset[label] = synth_dataset[label][:n]

    new_dataset = []
    for label in synth_dataset.keys():
        for label_text in synth_dataset[label]:
            new_dataset.append({label: label_text})
    return new_dataset, list(synth_dataset.keys())


def get_mscxr_synth_dataset_size_n(N, dataset, label_key="finding_labels", finding_key="label_text"):
    synth_dataset = []
    keys = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample[label_key]
        if str(label) not in DISEASES_TO_GENERATE:
            continue

        label = random.choice(label.split("|"))
        keys.add(label)

        label_texts = sample[finding_key]
        label_text = random.choice(label_texts.split("|"))
        synth_dataset.append({label: label_text})
    random.shuffle(synth_dataset)
    while len(synth_dataset) < N:
        logger.info(f"Artificially increasing dataset size by two because length of dataset is {len(synth_dataset)} but you requested {N}")
        synth_dataset = synth_dataset + synth_dataset
    synth_dataset = synth_dataset[:N]
    return synth_dataset, keys
