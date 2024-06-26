import os

from datasets import load_dataset, load_from_disk

from common_functions.functions import GetParentPath


def LoadDataTotal(config):
    parent_directory = GetParentPath(config, __file__)

    dir_to_save_total_data = config["preprocessing"]["dir_to_save_total_data"]

    save_folder = os.path.join(parent_directory, "data",
                               dir_to_save_total_data)

    if os.path.exists(save_folder):
        dataset = load_from_disk(save_folder)
        return dataset

    dataset = load_dataset('glue', 'sst2')
    dataset.save_to_disk(save_folder)

    return dataset


def LoadDataEach(config, type_dataset):
    dataset = LoadDataTotal(config)

    if type_dataset in ["train", "val"]:
        dataset = dataset["train"]
        train_test_split = dataset.train_test_split(test_size=0.1)

        type_return = "train" if type_dataset == "train" else "test"

        dataset = train_test_split[type_return]
    elif type_dataset == "test":
        dataset = dataset["validation"]
    else:
        raise ValueError(f"Type {type_dataset} doesn't exist")

    print(f"Retrieve dataset type {type_dataset} with"
          f"length of {len(dataset)}")

    return dataset
