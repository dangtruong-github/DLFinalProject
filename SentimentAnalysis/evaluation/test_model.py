import torch

import os

from train.models.encoder_no_attention.init_load_save import initNoAttention
from train.models.encoder_attention.init_load_save import initAttention
from train.train_total import load_model
from data_preprocessing.tokenize import GetTokenizer
from common_functions.functions import GetParentPath
from common_functions.constant import NOATTENTION, ATTENTION


def InferenceTest(config, text, name_model):
    print(name_model)
    parent_directory = GetParentPath(config, __file__)
    type_model = name_model.split("_")[0]

    if type_model == NOATTENTION:
        init_model = initNoAttention
    elif type_model == ATTENTION:
        init_model = initAttention

    tokenizer = GetTokenizer(config)

    max_token_length = int(config["preprocessing"]["max_token_length"])

    # Tokenize the input text
    inputs = tokenizer(text, padding="max_length", max_length=max_token_length,
                       return_tensors="pt")

    SAVE_FOLDER = os.path.join(parent_directory, "model_save", type_model,
                               name_model)
    model, _, optimizer = init_model(config)
    model, _, _ = load_model(model, optimizer, SAVE_FOLDER)

    print(inputs["input_ids"].shape)

    # Generate predictions
    with torch.no_grad():
        outputs = model(inputs["input_ids"].squeeze().unsqueeze(1))
        print(f"Shape of output: {outputs.shape}")
        preds = torch.argmax(outputs, dim=1)

    return preds, outputs
