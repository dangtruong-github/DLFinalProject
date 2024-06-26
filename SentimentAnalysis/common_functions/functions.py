import os
import json
import numpy as np


def GetParentPath(config, file, parent_folder_name=None):
    if parent_folder_name is None:
        parent_folder_name = config["general"]["containing_folder"]

    parent_directory = os.path.abspath(file)

    while os.path.basename(parent_directory) != parent_folder_name:
        parent_directory = os.path.dirname(parent_directory)

    return parent_directory


def GetDict(config):
    vn_filename_dict = config["preprocessing"]["vn_filename_dict"]
    en_filename_dict = config["preprocessing"]["en_filename_dict"]

    parent_directory = GetParentPath(config, __file__)

    vn_dict_path = os.path.join(parent_directory, "data", vn_filename_dict)
    en_dict_path = os.path.join(parent_directory, "data", en_filename_dict)

    with open(vn_dict_path, "r") as f:
        source_dict = json.load(f)

    with open(en_dict_path, "r") as f:
        target_dict = json.load(f)

    return source_dict, target_dict


def IndicesToSentence(
    np_indices: np.array,
    dict_words: dict
) -> str:
    print(np_indices[:5])
    sentence = ""
    rev_dict_words = {value: key for key, value in dict_words.items()}

    for i in range(np_indices.shape[0]):
        word = rev_dict_words[np_indices[i]]

        if word == "<pad>":
            break
        elif word == "</s>":
            break
        elif word == "<s>":
            continue
        elif word == "<unk>":
            sentence += "John "
            continue

        sentence += rev_dict_words[np_indices[i]]
        sentence += " "

    sentence = sentence[:-1]
    print(sentence)
    return sentence


def SentenceToIndices(
    sentence: str,
    dict_words: dict,
    max_indices: int
) -> np.array:
    indices = []
    tokens = sentence.split(" ")

    indices.append(dict_words["<s>"])

    for index, token in enumerate(tokens):
        if index >= max_indices - 1:
            break

        try:
            if token[0].isupper():
                token = token.capitalize()
        except Exception:
            return False

        if token in dict_words.keys():
            indices.append(dict_words[token])
        else:
            indices.append(dict_words["<unk>"])

    if len(indices) < max_indices:
        indices.append(dict_words["</s>"])

    while len(indices) < max_indices:
        indices.append(dict_words["<pad>"])

    np_indices = np.array(indices, dtype=np.int32)

    return np_indices
