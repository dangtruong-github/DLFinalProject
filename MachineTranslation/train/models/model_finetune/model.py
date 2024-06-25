from transformers import AutoModelForSeq2SeqLM

import os

from common_functions.functions import GetParentPath


def GetFineTuneModel(config):
    parent_directory = GetParentPath(config, __file__)
    model_folder = config["train"]["finetune_folder"]

    model_location = os.path.join(parent_directory, "model_save", model_folder)

    if os.path.exists(model_location):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_location)
        return model

    if os.path.exists(os.path.abspath(model_location)) is False:
        os.mkdir(os.path.abspath(model_location))

    model_name = "vinai/vinai-translate-en2vi"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.save_pretrained(model_location)

    return model
