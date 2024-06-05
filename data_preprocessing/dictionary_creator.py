import json
import os

from common_functions.functions import GetParentPath


def to_dict(
    file_read: str,
    file_path: str
):
    dict_words = {}
    dict_words["<pad>"] = 0
    dict_words["<eos>"] = 1
    dict_words["<sos>"] = 2
    dict_words["<unk>"] = 3
    index = 4

    with open(file_read, "r", encoding="utf8") as file:
        for line in file.readlines():
            sentence = line[:-1]

            tokens = sentence.split(" ")

            for token in tokens:
                token = token.capitalize()
                if token not in dict_words.keys():
                    dict_words[token] = index
                    index += 1

        print(f"Length of json index: {index}")

    with open(file_path, 'w') as f:
        json.dump(dict_words, f)


def CreateDictionary(config):
    parent_folder_name = config["general"]["containing_folder"]

    vn_filename_train = config["preprocessing"]["vn_filename_train"]
    en_filename_train = config["preprocessing"]["en_filename_train"]
    vn_filename_dict = config["preprocessing"]["vn_filename_dict"]
    en_filename_dict = config["preprocessing"]["en_filename_dict"]

    parent_directory = GetParentPath(parent_folder_name, __file__)

    to_save_path = os.path.join(parent_directory, "data")

    vn_filename_train_total = os.path.join(to_save_path, vn_filename_train)
    en_filename_train_total = os.path.join(to_save_path, en_filename_train)
    vn_filename_dict_total = os.path.join(to_save_path, vn_filename_dict)
    en_filename_dict_total = os.path.join(to_save_path, en_filename_dict)

    to_dict(vn_filename_train_total, vn_filename_dict_total)
    to_dict(en_filename_train_total, en_filename_dict_total)
