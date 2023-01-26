from .cars import CarDataset
from .birds import BirdDataset
from .human import HumanDataset
from .dogs import DogDataset
from .chest import ChestXray14Dataset, ChestXray14BboxDataset, MimicCXRDataset


def get_dataset(opt, split=None):
    datasets = {"bird":BirdDataset, "human36": HumanDataset, "dog":DogDataset, "car": CarDataset, "chestxray14": ChestXray14Dataset, "chestxray14bbox":ChestXray14BboxDataset,
                "chestxraymimic": MimicCXRDataset}
    assert opt.dataset in datasets.keys(), f"Dataset has to be one of: {datasets.keys()}"

    if split is not None:
        if split == "train":
            opt.dataset_args = opt.dataset_args_train
        elif split == "val":
            opt.dataset_args = opt.dataset_args_val
        elif split == "test":
            opt.dataset_args = opt.dataset_args_test
        else:
            raise ValueError("dataset has to be one of (train, test, val)")

    dataset = datasets[opt.dataset](opt, opt.H, opt.W, mask_dir=opt.out_dir)
    return dataset