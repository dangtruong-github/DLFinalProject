import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple
import numpy as np

from train.models.encoder_no_attention.model import EncoderClassification
from data_preprocessing.tokenize import GetTokenizer, LoadHfTokenized

device = "cuda" if torch.cuda.is_available() else "cpu"


def initNoAttention(
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
    embedding_size = int(config["encoder_no_attention"]["embeddings_size"])
    hidden_size = int(config["encoder_no_attention"]["hidden_size"])
    num_layers = int(config["encoder_no_attention"]["num_layers"])
    drop_out = float(config["encoder_no_attention"]["dropout_rate"])

    vocab_dict = GetTokenizer(config).get_vocab()

    dataset = LoadHfTokenized(config, type_dataset="train")
    tmp_data = np.array(dataset["label"])
    num_classes = np.max(tmp_data) + 1
    tmp_data = None
    dataset = None

    model = EncoderClassification(len(vocab_dict.keys()),
                                  embedding_size,
                                  hidden_size,
                                  num_layers,
                                  drop_out,
                                  num_classes).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # print(type(model))
    # print(type(criterion))
    # print(type(optimizer))

    return model, criterion, optimizer
