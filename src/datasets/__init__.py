from .cars import CarDataset
from .birds import BirdDataset
from .human import HumanDataset
from .dogs import DogDataset
from .chest import ChestXray14Dataset, ChestXray14BboxDataset, MimicCXRDataset, MimicCXRDatasetMSBBOX
from copy import deepcopy


def get_dataset(opt, split=None):
    datasets = {"chestxray14": ChestXray14Dataset, "chestxray14bbox":ChestXray14BboxDataset, "chestxraymimic": MimicCXRDataset, "chestxraymimicbbox": MimicCXRDatasetMSBBOX}
    assert split is not None
    dataset_args = getattr(opt, f"dataset_args_{split}")
    getattr(opt, "dataset_args", dataset_args)
    dataset = datasets[dataset_args["dataset"]](dataset_args=dataset_args, opt=opt)
    return dataset