import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from common_functions.functions import GetParentPath


class CustomDataset(Dataset):
    def __init__(self, config, type_dataset):
        parent_folder_name = config["general"]["containing_folder"]
        key_file_name_to_save = "filename_to_save_" + type_dataset
        file_name_to_save = config["preprocessing"][key_file_name_to_save]

        parent_directory = GetParentPath(parent_folder_name, __file__)
        data_path = os.path.join(parent_directory, "data", file_name_to_save)

        self.data = np.load(data_path)
        self.total_len = self.data.shape[0]

        self.vn_max_indices = int(config["preprocessing"]["vn_max_indices"])

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        vn_indices = self.data[index][:self.vn_max_indices]
        en_indices = self.data[index][self.vn_max_indices:]

        torch_vn_indices = torch.tensor(vn_indices, dtype=torch.int64)
        torch_en_indices = torch.tensor(en_indices, dtype=torch.int64)

        return (torch_vn_indices, torch_en_indices)


def CustomLoader(
    config,
    type_dataset: str
) -> DataLoader:
    if type_dataset not in ["train", "val", "test"]:
        raise ValueError(f"Type dataset {type_dataset} does not exist")

    custom_set = CustomDataset(config, type_dataset)

    batch_size = int(config["train"]["batch_size"])
    shuffle = type_dataset == "train"

    custom_loader = DataLoader(custom_set, batch_size, shuffle)

    print(f"Success creating data loader of {type_dataset}")

    return custom_loader
