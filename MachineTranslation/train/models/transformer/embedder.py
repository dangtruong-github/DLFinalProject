import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(Embedder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = d_model

        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        out = self.embed(x)
        return out
