from torch.utils.data import DataLoader

from data_preprocessing.dictionary_creator import CreateDictionary
from data_preprocessing.index_converter import FileToIndices
from data_preprocessing.loader import CustomLoader


def ProcessingPipeline(config):
    CreateDictionary(config)
    FileToIndices(config, type_dataset="train")
    FileToIndices(config, type_dataset="val")
    FileToIndices(config, type_dataset="test")


def LoaderPipeline(
    config,
    type_dataset
) -> DataLoader:
    loader = CustomLoader(config, type_dataset)
    return loader
