import os
import pandas as pd

from common_functions.functions import GetParentPath


def LoadDataEach(config, en_path, vn_path):
    parent_directory = GetParentPath(config, __file__)

    path_data_en = os.path.join(parent_directory, en_path)
    path_data_vn = os.path.join(parent_directory, vn_path)

    with open(path_data_en, "r") as file:
        text = file.readlines()
        text = [line.strip() for line in text]
        df_en = pd.DataFrame(text, columns=["english"])

    with open(path_data_vn, "r") as file:
        text = file.readlines()
        text = [line.strip() for line in text]
        df_vn = pd.DataFrame(text, columns=["vietnamese"])

    df = pd.concat([df_vn, df_en], axis=1)

    return df


def TextToDf(config, type_dataset):
    key = "filename_" + type_dataset
    filename = config["preprocessing"]["{}".format(key)]

    vn_filename = "{}.vi".format(filename)
    en_filename = "{}.en".format(filename)

    df = LoadDataEach(config, en_path=en_filename, vn_path=vn_filename)

    return df
