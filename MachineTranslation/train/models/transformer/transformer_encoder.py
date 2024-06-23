import torch.nn as nn

from train.models.transformer.embedder import Embedder
from train.models.transformer.positional_encoder import PositionalEncoder
from train.models.transformer.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, d_model=512,
                 num_layer=6, factor=4, n_head=8):
        super(TransformerEncoder, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.factor = factor
        self.n_head = n_head
        self.d_model = d_model

        self.embedding_layer = Embedder(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoder = PositionalEncoder(seq_len=seq_len,
                                                    d_model=d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model,
                              n_head=n_head,
                              factor=factor) for i in range(num_layer)]
        )

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        return out
