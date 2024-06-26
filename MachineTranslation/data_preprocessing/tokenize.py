from transformers import AutoTokenizer
from datasets import Dataset

import os

from common_functions.functions import GetParentPath

source_lang = "english"
target_lang = "vietnamese"

checkpoint = "vinai/phobert-base"


def GetTokenizer(config):
    parent_directory = GetParentPath(config, __file__)
    print(parent_directory)
    tokenizer_folder_name = config["preprocessing"]["tokenizer_folder"]

    tokenizer_location = os.path.join(parent_directory, "data",
                                      tokenizer_folder_name)

    if os.path.exists(tokenizer_location):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)
        print("Success loading saved tokenizer")
        return tokenizer
    else:
        os.makedirs(tokenizer_location, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # new_special_tokens = {"additional_special_tokens": ["<sos>", "<eos>"]}

    # Add the new special token to the tokenizer
    # tokenizer.add_special_tokens(new_special_tokens)

    tokenizer.save_pretrained(tokenizer_location)
    print(f"Success loading pretrained tokenizer from {checkpoint}")

    return tokenizer


def LoadHfTokenized(config, type_dataset):
    """
    Return dataset loaded if exists, else return False
    """
    parent_directory = GetParentPath(config, __file__)

    dir_to_load_hf_data = config["preprocessing"]["dir_to_save_hf_data"]
    dir_to_load = "{}_{}".format(dir_to_load_hf_data, type_dataset)

    dir_to_load_final = os.path.join(parent_directory, "data", dir_to_load)

    if os.path.exists(dir_to_load_final) is False:
        return False

    hf_val_tokenized_loaded = Dataset.load_from_disk(dir_to_load_final)

    print(f"Load {type_dataset} success in {dir_to_load_final}")

    return hf_val_tokenized_loaded


def SaveHfTokenized(config, hf_dataset_tokenized, type_dataset):
    parent_directory = GetParentPath(config, __file__)

    dir_to_save_hf_data = config["preprocessing"]["dir_to_save_hf_data"]
    dir_to_save = "{}_{}".format(dir_to_save_hf_data, type_dataset)

    dir_to_save_final = os.path.join(parent_directory, "data", dir_to_save)
    print(f"Directory to save: {dir_to_save_final}")

    if os.path.exists(dir_to_save_final) is False:
        os.makedirs(dir_to_save_final, exist_ok=True)

    hf_dataset_tokenized.save_to_disk(dir_to_save_final)

    print(f"Save {type_dataset} success to {dir_to_save_final}")


def TokenizeHfDataset(config, hf_dataset, save, type_dataset):
    max_token_length = int(config["preprocessing"]["max_token_length"])

    def preprocess_function(examples):
        # print(examples)

        inputs = [example for example in examples[source_lang]]
        targets = [example for example in examples[target_lang]]

        tokenizer = GetTokenizer(config)

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
