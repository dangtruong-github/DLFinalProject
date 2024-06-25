import torch
import torch.nn as nn

import random

device = "cuda" if torch.cuda.is_available() else "cpu"


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rev_words_dict):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.rev_words_dict = rev_words_dict
        self.vocab_len = len(rev_words_dict.keys())

    def forward(self, source, target=None, teacher_force_ratio=.5):
        # input shape (seq_len, batch_size)
        target_len = source.shape[0]
        batch_size = source.shape[1]

        # print(f"Target len: {target_len}, batch_size = {batch_size}")

        outputs = torch.zeros(target_len,
                              batch_size,
                              self.vocab_len).to(device=device)
        outputs[0, :, 1] = torch.tensor(1)

        hidden, cell = self.encoder(source)

        # print(f"Hidden size: {hidden.shape}, cell size = {cell.shape}")

        x = torch.ones(batch_size, dtype=torch.int64)

        if target is not None:
            x = target[0]

        # print(f"x shape: {x.shape} of {x}")

        for i in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[i] = output

            best_guess = output.argmax(1)

            if target is None:
                x = best_guess
            else:
                if random.random() < teacher_force_ratio:
                    x = x = target[i]
                else:
                    x = best_guess

        return outputs
