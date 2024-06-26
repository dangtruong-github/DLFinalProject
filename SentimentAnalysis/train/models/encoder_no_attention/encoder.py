import torch.nn as nn


def init_lstm_weights(lstm, low=-0.08, high=0.08):
    for name, param in lstm.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, low, high)
        elif 'bias' in name:
            nn.init.zeros_(param)

        return lstm


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, num_layers, drop_out):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drop_out)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_size,
                                       padding_idx=0,
                                       max_norm=None,
                                       norm_type=2.0,
                                       scale_grad_by_freq=False,
                                       sparse=False)

        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=drop_out)

        self.rnn = init_lstm_weights(self.rnn)

    def forward(self, x):
        # x: input shape (seq_max_len, batch_size)
        # print(x.shape)
        embeddings_pre_dropout = self.embeddings(x)

        # print(embeddings_pre_dropout.shape)

        # print(embeddings_pre_dropout.is_leaf)
        # print(x.is_leaf)
        # print(embeddings_pre_dropout.mean().backward())

        embeddings = self.dropout(embeddings_pre_dropout)
        # print(embeddings.is_leaf)

        output, (hidden, cell) = self.rnn(embeddings)

        return hidden, cell
