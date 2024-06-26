import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple
import numpy as np

from train.models.encoder_attention.classification_model import (
    ClassificationModel
)
from data_preprocessing.tokenize import GetTokenizer, LoadHfTokenized

device = "cuda" if torch.cuda.is_available() else "cpu"


def initAttention(
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
    d_model = int(config["encoder_attention"]["d_model"])
    num_layers = int(config["encoder_attention"]["num_layers"])
    max_token_length = int(config["preprocessing"]["max_token_length"])
    n_head = int(config["encoder_attention"]["n_head"])
    d_ff = int(config["encoder_attention"]["d_ff"])
    dropout = float(config["encoder_attention"]["dropout"])

    vocab_dict = GetTokenizer(config).get_vocab()

    dataset = LoadHfTokenized(config, type_dataset="train")
    tmp_data = np.array(dataset["label"])
    num_classes = np.max(tmp_data) + 1
    tmp_data = None
    dataset = None

    # print(f"Vocab size: {len(vocab_dict.keys())}")
    # print(f"d_model: {d_model}")

    model = ClassificationModel(vocab_size=len(vocab_dict.keys()),
                                d_model=d_model,
                                n_head=n_head,
                                num_layers=num_layers,
                                d_ff=d_ff,
                                max_seq_length=max_token_length,
                                dropout=dropout,
                                num_classes=num_classes,
                                config=config).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # print(type(model))
    # print(type(criterion))
    # print(type(optimizer))

    return model, criterion, optimizer
