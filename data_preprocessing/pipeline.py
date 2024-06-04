from .dictionary_creator import CreateDictionary
from .index_converter import FileToIndices


def ProcessingPipeline(config):
    CreateDictionary(config)
    FileToIndices(config, type_dataset="train")
    FileToIndices(config, type_dataset="val")
    FileToIndices(config, type_dataset="test")


def LoaderPipeline(config, type_dataset):
    pass
