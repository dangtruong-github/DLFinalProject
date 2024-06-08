import torch.nn as nn

from train.models.transformer.embedder import Embedder
from train.models.transformer.positional_encoder import PositionalEncoder
from train.models.transformer.decoder_block import DecoderBlock


class TransformerDecoder(nn.Module):
    def __init__(self, seq_len, vocab_size, d_model=512,
                 num_layer=2, factor=4, n_head=8):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layer = num_layer
        self.factor = factor
        self.n_head = n_head

        self.embedding_layer = Embedder(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoder = PositionalEncoder(seq_len=seq_len,
                                                    d_model=d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, factor=factor, n_head=n_head)
            for i in range(self.num_layer)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, encoder_out, mask):
        x = self.embedding_layer(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x=x, k=encoder_out, v=encoder_out, mask=mask)

        out = nn.functional.softmax(self.fc_out(x), dim=-1)
        # (batch_size, seq_len, vocab_size)

        return out
