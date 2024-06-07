import os
import numpy as np
import gc

from common_functions.functions import GetParentPath, GetDict
from common_functions.functions import SentenceToIndices


def ConcatVnEnIndices(
    vn_np_indices: np.array,
    en_np_indices: np.array
) -> np.array:
    concat_indices = np.concatenate((vn_np_indices, en_np_indices),
                                    dtype=np.int32)
    concat_indices = np.expand_dims(concat_indices, axis=0)
    return concat_indices


def FileToIndices(
    config,
    type_dataset: str,
    window: int = 700
):
    if type_dataset not in ["train", "val", "test"]:
        raise Exception(f"Type dataset {type_dataset} does not exist")

    parent_folder_name = config["general"]["containing_folder"]

    vn_filename = config["preprocessing"]["vn_filename_" + type_dataset]
    en_filename = config["preprocessing"]["en_filename_" + type_dataset]

    key_filename_to_save = "filename_to_save_" + type_dataset
    key_filename_to_save_noise = "filename_to_save_noise_" + type_dataset
    filename_to_save = config["preprocessing"][key_filename_to_save]
    filename_save_noise = config["preprocessing"][key_filename_to_save_noise]
    vn_max_indices = int(config["preprocessing"]["vn_max_indices"])
    en_max_indices = int(config["preprocessing"]["vn_max_indices"])

    parent_directory = GetParentPath(parent_folder_name, __file__)

    vn_filename_path = os.path.join(parent_directory, "data", vn_filename)
    en_filename_path = os.path.join(parent_directory, "data", en_filename)
    filename_to_save_path = os.path.join(parent_directory, "data",
                                         filename_to_save)
    filename_to_save_noise_path = os.path.join(parent_directory, "data",
                                               filename_save_noise)

    vn_dict, en_dict = GetDict(config)

    total_indices = None
    tmp_indices = None
    cur_index = 0

    noise_indices = []
    noise_indices_tmp = []

    if os.path.exists(filename_to_save_path):
        total_indices = np.load(filename_to_save_path)
        noise_indices = np.load(filename_to_save_noise_path)
        noise_indices = noise_indices.tolist()

        cur_index = total_indices.shape[0] + len(noise_indices)
        print(f"Cached indices shape: {cur_index}")
        total_indices = None
        noise_indices = []

    with open(vn_filename_path, "r", encoding='utf8') as vn_file:
        with open(en_filename_path, "r", encoding='utf8') as en_file:
            en_data = en_file.readlines()
            for index, vn_row in enumerate(vn_file.readlines()):
                if index < cur_index:
                    continue

                vn_row = vn_row[:-1]
                en_row = en_data[index][:-1]

                vn_np_indices = SentenceToIndices(vn_row, vn_dict,
                                                  vn_max_indices)
                en_np_indices = SentenceToIndices(en_row, en_dict,
                                                  en_max_indices)

                if isinstance(vn_np_indices, bool) or isinstance(en_np_indices,
                                                                 bool):
                    print(f"Noise index: {index}")
                    noise_indices_tmp.append(index)
                    print(f"VN sentence: {vn_row}")
                    print(f"EN sentence: {en_row}")
                    continue

                cur_indices = ConcatVnEnIndices(vn_np_indices, en_np_indices)

                if tmp_indices is None:
                    tmp_indices = cur_indices
                else:
                    tmp_indices = np.concatenate([tmp_indices, cur_indices],
                                                 axis=0, dtype=np.int32)

                if index % 1000 == 0:
                    print(f"Finish index {index}")

                if tmp_indices.shape[0] >= window:
                    # tmp_indices
                    if os.path.exists(filename_to_save_path):
                        total_indices = np.load(filename_to_save_path)

                        total_indices = np.concatenate(
                            (total_indices, tmp_indices),
                            axis=0, dtype=np.int32)
                    else:
                        total_indices = tmp_indices

                    np.save(filename_to_save_path, total_indices)
                    print(f"Current shape {total_indices.shape}")

                    total_indices = None
                    tmp_indices = None

                    gc.collect()

                    # noise_indices
                    if os.path.exists(filename_to_save_noise_path):
                        noise_indices = np.load(filename_to_save_noise_path)
                        noise_indices = noise_indices.tolist()
                        noise_indices.extend(noise_indices_tmp)
                    else:
                        noise_indices = noise_indices_tmp

                    noise_indices = np.array(noise_indices)

                    np.save(filename_to_save_noise_path, noise_indices)
                    print(f"Current noise indices size {noise_indices.shape}")

                    noise_indices = None
                    noise_indices_tmp = []

                    print(f"Current in index {index}")
                    print("Success")

    if tmp_indices is not None:
        if os.path.exists(filename_to_save_path):
            total_indices = np.load(filename_to_save_path)

            total_indices = np.concatenate(
                (total_indices, tmp_indices),
                axis=0, dtype=np.int32)
        else:
            total_indices = tmp_indices

        np.save(filename_to_save_path, total_indices)
        print(f"Final shape {total_indices.shape}")

        total_indices = None
        tmp_indices = None

        gc.collect()

    print(f"Total noise indices: {noise_indices}")

    print("Success converting to indices")
