import torch
import torch.nn as nn

import math

from train.models.encoder_attention.encoder_layer import (
    EncoderLayer
)
from train.models.encoder_attention.positional_encoding import (
    PositionalEncoding
)
from data_preprocessing.tokenize import GetTokenizer


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers,
                 d_ff, max_seq_length, dropout, config):
        super(TransformerEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.positional_embedding = PositionalEncoding(d_model, max_seq_length)

        tokenizer = GetTokenizer(config)
        self.pad_token_id = tokenizer.pad_token_id

    def masking(self, x):
        x_mask = (x != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        x_mask = x_mask.to(x.device)
        return x_mask

    def forward(self, x):
        x_mask = self.masking(x)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(self.positional_embedding(x))

        for layer in self.encoder:
            x = layer(x, x_mask)

        x = x.reshape(x.shape[1], -1)

        return x


if __name__ == "__main__":
    x = torch.randint(size=(32, 10), low=0, high=1000)

    net = TransformerEncoder(vocab_size=1000, d_model=512, n_head=8,
                             num_layers=4, d_ff=2048, max_seq_length=10,
                             dropout=0.1)

    print(net(x).shape)
