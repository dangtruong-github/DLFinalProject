import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data["input_ids"]
        self.label = data["labels"]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        data = torch.tensor(data, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)

        return (data, label)


def CustomLoader(
    config,
    hf_dataset_tokenized,
    type_dataset
) -> DataLoader:
    custom_set = CustomDataset(config, hf_dataset_tokenized)

    testing_mode = bool(config["general"]["test"])
    batch_size = int(config["train"]["batch_size"])

    if testing_mode:
        batch_size = int(config["train"]["batch_size_test"])

    shuffle = type_dataset == "train"

    custom_loader = DataLoader(custom_set, batch_size, shuffle)

    print(f"Success creating data loader of {type_dataset}")

    return custom_loader
