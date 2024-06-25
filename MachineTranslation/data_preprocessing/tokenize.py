from transformers import AutoTokenizer
from datasets import DatasetDict

import os

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "english"
target_lang = "vietnamese"


def LoadHfTokenized(config, type_dataset):
    """
    Return dataset loaded if exists, else return False
    """
    parent_folder_name = config["general"]["containing_folder"]

    dir_to_load_hf_data = config["preprocessing"]["dir_to_save_hf_data"]
    dir_to_load = "{}_{}".format(dir_to_load_hf_data, type_dataset)

    dir_to_load_final = os.path.join(parent_folder_name, "data", dir_to_load)

    if os.exists(dir_to_load_final) is False:
        return False

    hf_val_tokenized_loaded = DatasetDict.load_from_disk(dir_to_load_final)

    print(f"Load {type_dataset} success in {dir_to_load_final}")

    return hf_val_tokenized_loaded


def SaveHfTokenized(config, hf_dataset_tokenized, type_dataset):
    parent_folder_name = config["general"]["containing_folder"]

    dir_to_save_hf_data = config["preprocessing"]["dir_to_save_hf_data"]
    dir_to_save = "{}_{}".format(dir_to_save_hf_data, type_dataset)

    dir_to_save_final = os.path.join(parent_folder_name, "data", dir_to_save)

    hf_dataset_tokenized.save_to_disk(dir_to_save_final)

    print(f"Save {type_dataset} success to {dir_to_save_final}")


def TokenizeHfDataset(config, hf_dataset, save, type_dataset):
    max_token_length = config["preprocessing"]["max_token_length"]

    def preprocess_function(examples):
        # print(examples)

        inputs = [example for example in examples[source_lang]]
        targets = [example for example in examples[target_lang]]

        model_inputs = tokenizer(inputs,
                                 text_target=targets,
                                 padding="max_length",
                                 max_length=max_token_length,
                                 truncation=True)

        return model_inputs

    hf_dataset_tokenized = hf_dataset.map(preprocess_function, batched=True)

    if save:
        SaveHfTokenized(config, hf_dataset_tokenized, type_dataset)

    return hf_dataset_tokenized
