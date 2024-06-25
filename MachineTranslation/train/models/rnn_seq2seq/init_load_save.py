import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

from train.models.rnn_seq2seq.model import Seq2Seq
from train.models.rnn_seq2seq.encoder_decoder import Encoder, Decoder
from data_preprocessing.tokenize import GetTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def initSeq2Seq(
    config
) -> Tuple[
    nn.Module,
    nn.modules.loss._Loss,
    optim.Optimizer
]:
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

    vocab_dict = GetTokenizer(config).get_vocab()

    encoder = Encoder(len(vocab_dict.keys()),
                      embeddings_size,
                      hidden_size,
                      num_layers,
                      dropout_rate)
    decoder = Decoder(len(vocab_dict.keys()),
                      embeddings_size,
                      hidden_size,
                      num_layers,
                      dropout_rate)

    model = Seq2Seq(encoder, decoder, vocab_dict).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # print(type(model))
    # print(type(criterion))
    # print(type(optimizer))

    return model, criterion, optimizer
