from .chest import ChestXray14Dataset, ChestXray14BboxDataset, MimicCXRDataset, MimicCXRDatasetMSBBOX
from .mscoco import MSCOCODataset, MSCOCOBBoxDataset
from copy import deepcopy


def get_dataset(opt, split=None):
    datasets = {"chestxray14": ChestXray14Dataset, "chestxray14bbox":ChestXray14BboxDataset, "chestxraymimic": MimicCXRDataset, "chestxraymimicbbox": MimicCXRDatasetMSBBOX, "mscoco": MSCOCODataset, "mscocobbox": MSCOCOBBoxDataset}
    assert split is not None
    dataset_args = getattr(opt.datasets, f"{split}")
    getattr(opt, "dataset_args", dataset_args)
    dataset = datasets[dataset_args["dataset"]](dataset_args=dataset_args, opt=opt)
    return dataset