from torch.utils.data import DataLoader

from data_preprocessing.load_data import LoadDataEach
from data_preprocessing.tokenize import TokenizeHfDataset, LoadHfTokenized
from data_preprocessing.loader import CustomLoader


def ProcessingPipeline(config, type_dataset, save):
    hf_dataset_tokenized_loaded = LoadHfTokenized(config, type_dataset)

    if isinstance(hf_dataset_tokenized_loaded, bool) is False:
        print(f"Success finding cache dataset type {type_dataset}")
        return hf_dataset_tokenized_loaded

    hf_dataset = LoadDataEach(config, type_dataset)
    hf_dataset_tokenized = TokenizeHfDataset(config, hf_dataset, save,
                                             type_dataset)

    return hf_dataset_tokenized


def LoaderPipeline(
    config,
    type_dataset,
    save=True
) -> DataLoader:
    if type_dataset not in ["train", "val", "test"]:
        raise ValueError(f"Type dataset {type_dataset} does not exist")

    hf_dataset_tokenized = ProcessingPipeline(config, type_dataset, save)
    loader = CustomLoader(config, hf_dataset_tokenized, type_dataset)
    return loader
