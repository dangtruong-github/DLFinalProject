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
        self.word_dict = {value: key for key, value in rev_words_dict.items()}
        self.vocab_len = len(rev_words_dict.keys())

    def forward(self, source, target, teacher_force_ratio=.5):
        # input shape (seq_len, batch_size)
        target_len = target.shape[0]
        batch_size = target.shape[1]

        # print(f"Target len: {target_len}, batch_size = {batch_size}")

        outputs = torch.zeros(target_len,
                              batch_size,
                              self.vocab_len).to(device=device)
        outputs[0, :, self.word_dict["<sos>"]] = torch.tensor(1)

        hidden, cell = self.encoder(source)

        x = target[0]

        print(x)

        for i in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[i] = output

            best_guess = output.argmax(1)

            rando = random.random() < teacher_force_ratio
            x = target[i] if rando else best_guess

        # print(f"Output inside seq2seq forward: {outputs[:5]}")

        return outputs

    def predict(self, source, max_len=50):
        outputs = torch.zeros(max_len, dtype=torch.int64)

        hidden, cell = self.encoder(source)

        x = torch.tensor(self.word_dict["<sos>"], dtype=torch.int64)

        outputs[0] = x

        current_length = 1

        for i in range(1, max_len):
            current_length += 1

            output, hidden, cell = self.decoder(x, hidden, cell)

            best_guess = output.argmax(0)

            outputs[i] = best_guess

            if self.rev_words_dict[int(best_guess)] in ["<eos>", "<pad>"]:
                break

            x = best_guess

            print(self.rev_words_dict[int(best_guess)], end=" ")

        return outputs[:current_length]
