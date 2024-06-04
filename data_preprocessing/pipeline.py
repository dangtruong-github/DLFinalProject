from .dictionary_creator import CreateDictionary
from .index_converter import FileToIndices


def ProcessingPipeLine(config):
    # CreateDictionary(config)
    FileToIndices(config, type_dataset="train")
