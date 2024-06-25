import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

from train.models.transformer.transformer_total import Transformer
from data_preprocessing.tokenize import GetTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def initTransformer(
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
    d_model = int(config["transformer"]["d_model"])
    num_layers = int(config["transformer"]["num_layers"])
    seq_len = int(config["preprocessing"]["vn_max_indices"])

    vocab_dict = GetTokenizer(config).get_vocab()

    model = Transformer(d_model=d_model,
                        src_vocab_size=len(vocab_dict.keys()),
                        target_vocab_size=len(vocab_dict.keys()),
                        seq_len=seq_len,
                        num_layer=num_layers).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # print(type(model))
    # print(type(criterion))
    # print(type(optimizer))

    return model, criterion, optimizer
