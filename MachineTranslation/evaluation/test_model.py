import torch

import os

from train.models.rnn_seq2seq.init_load_save import initSeq2Seq
from train.models.transformer.init_load_save import initTransformer
from train.train_total import load_model
from data_preprocessing.tokenize import GetTokenizer
from common_functions.functions import GetParentPath
from common_functions.constant import SEQ2SEQ, TRANSFORMER


def InferenceTest(config, text, name_model):
    print(name_model)
    parent_directory = GetParentPath(config, __file__)
    type_model = name_model.split("_")[0]

    if type_model == SEQ2SEQ:
        init_model = initSeq2Seq
    elif type_model == TRANSFORMER:
        init_model = initTransformer

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
        preds = torch.argmax(outputs, dim=2)
    decoded_preds = tokenizer.decode(preds[0], skip_special_tokens=True)
    return decoded_preds
