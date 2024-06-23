import torch
import torch.nn as nn

from train.models.transformer.transformer_encoder import TransformerEncoder
from train.models.transformer.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, d_model, src_vocab_size, target_vocab_size, seq_len,
                 num_layer=6, factor=4, n_head=8):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        self.seq_len = seq_len
        self.num_laye = num_layer
        self.factor = factor
        self.n_head = n_head

        self.encoder = TransformerEncoder(seq_len=seq_len,
                                          vocab_size=src_vocab_size,
                                          d_model=d_model,
                                          num_layer=num_layer,
                                          factor=factor,
                                          n_head=n_head)
        self.decoder = TransformerDecoder(seq_len=seq_len,
                                          vocab_size=target_vocab_size,
                                          d_model=d_model,
                                          num_layer=num_layer,
                                          factor=factor,
                                          n_head=n_head)

    def make_target_mask(self, trg):
        device = trg.device
        tgt_mask = (trg != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = trg.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length),
                                      diagonal=1))
        nopeak_mask = nopeak_mask.bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask.to(trg.device)

    def decode(self, src):
        batch_size, seq_len = src.shape[0], src.shape[1]
        trg = torch.zeros(size=(batch_size, seq_len), dtype=torch.int)
        trg[:, 0] = 1

        enc_out = self.encoder(src)

        for i in range(1, seq_len):
            trg_mask = self.make_target_mask(trg)
            out = self.decoder(x=trg, encoder_out=enc_out, mask=trg_mask)
            out = out.argmax(-1)[:, i]
            trg[:, i] = out

        return trg

    def forward(self, src, trg):

        trg_mask = self.make_target_mask(trg)
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs
