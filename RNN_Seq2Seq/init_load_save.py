import torch
import torch.nn as nn
import torch.optim as optim

from model import Seq2Seq
from encoder_decoder import Encoder, Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def initSeq2Seq(config, source_dict, target_dict):
    """
    Args:
    - config: get from config.ini file
    - source_dict: dictionary of words of the source language
    - target_dict: dictionary of words of the target language
    Output:
    - model: Seq2Seq model
    - criterion: CrossEntropyLoss
    - optimizer: Adam
    """
    learning_rate = config["train"]["embeddings_size"]
    weight_decay = config["train"]["weight_decay"]
    embeddings_size = config["rnn_seq2seq"]["embeddings_size"]
    hidden_size = config["rnn_seq2seq"]["hidden_size"]
    num_layers = config["rnn_seq2seq"]["num_layers"]
    dropout_rate = config["rnn_seq2seq"]["dropout_rate"]

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

    return model, criterion, optimizer
