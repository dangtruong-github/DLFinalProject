import torch
import torch.nn as nn

from train.models.encoder_no_attention.encoder import Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class EncoderClassification(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 num_layers, drop_out, num_classes):
        super(EncoderClassification, self).__init__()

        self.encoder = Encoder(vocab_size, embedding_size, hidden_size,
                               num_layers, drop_out)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, cell = self.encoder(x)

        x = cell[-1::].squeeze(0)

        x = self.fc1(x)

        return x
