import torch
import torch.nn as nn
import torch.optim as optim

import os
import json

from .model import Seq2Seq
from .encoder_decoder import Encoder, Decoder
from common_functions.functions import GetParentPath

device = "cuda" if torch.cuda.is_available() else "cpu"


def initSeq2Seq(config):
    """
    Args:
    - config: get from config.ini file
    Output:
    - model: Seq2Seq model
    - criterion: CrossEntropyLoss
    - optimizer: Adam
    """
    learning_rate = float(config["train"]["learning_rate"])
    weight_decay = float(config["train"]["weight_decay"])
    embeddings_size = int(config["rnn_seq2seq"]["embeddings_size"])
    hidden_size = int(config["rnn_seq2seq"]["hidden_size"])
    num_layers = int(config["rnn_seq2seq"]["num_layers"])
    dropout_rate = float(config["rnn_seq2seq"]["dropout_rate"])

    vn_filename_dict = config["preprocessing"]["vn_filename_dict"]
    en_filename_dict = config["preprocessing"]["en_filename_dict"]

    parent_folder_name = config["general"]["containing_folder"]
    parent_directory = GetParentPath(parent_folder_name, __file__)

    vn_dict_path = os.path.join(parent_directory, "data", vn_filename_dict)
    en_dict_path = os.path.join(parent_directory, "data", en_filename_dict)

    with open(vn_dict_path, "r") as f:
        source_dict = json.load(f)

    with open(en_dict_path, "r") as f:
        target_dict = json.load(f)

    encoder = Encoder(len(source_dict.keys()),
                      embeddings_size,
                      hidden_size,
                      num_layers,
                      dropout_rate)
    decoder = Decoder(len(target_dict.keys()),
                      embeddings_size,
                      hidden_size,
                      num_layers,
                      dropout_rate)

    rev_target_dict = {value: key for key, value in target_dict.items()}

    model = Seq2Seq(encoder, decoder, rev_target_dict).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    print(type(model))
    print(type(criterion))
    print(type(optimizer))

    return model, criterion, optimizer
