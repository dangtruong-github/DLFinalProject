import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import configparser

from train.models.encoder_attention.transformer_encoder import (
    TransformerEncoder
)


class ClassificationModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, n_head=8,
                 num_layers=4, d_ff=2048, max_seq_length=64,
                 dropout=0.1, num_classes=2, config=None):
        super(ClassificationModel, self).__init__()
        self.transformers_encoder = TransformerEncoder(vocab_size, d_model,
                                                       n_head, num_layers,
                                                       d_ff, max_seq_length,
                                                       dropout, config)

        self.fc1 = nn.Linear(max_seq_length * d_model, d_model)
        self.fc2 = nn.Linear(d_model, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.transformers_encoder(x)

        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Create a ConfigParser instance
    config = configparser.ConfigParser()

    parent_directory = os.path.abspath(__file__)

    for i in range(4):
        parent_directory = os.path.dirname(parent_directory)

    print(parent_directory)

    # Read your configuration file (if needed)
    config.read(os.path.join(parent_directory, "config.ini"))

    x = torch.randint(size=(32, 10), low=1, high=100)

    net = ClassificationModel(vocab_size=1000, d_model=512, n_head=8,
                              num_layers=4, d_ff=2048, max_seq_length=10,
                              dropout=0.1, num_classes=2, config=config)
    a = net(x)
    print(a.shape)
