import torch.nn as nn

from train.models.transformer.multihead_attention import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_head=8, factor=4):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.factor = factor

        self.multihead_attention = MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, factor*d_model),
            nn.ReLU(),
            nn.Linear(factor*d_model, d_model)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, q, k, v):
        """
        x: (batch_size, sequence length, embedded dimension)
        """

        attention_out = self.multihead_attention(q, k, v)
        attention_res_out = attention_out + v
        norm1_out = self.dropout1(self.norm1(attention_res_out))
        fw_out = self.feed_forward(norm1_out)
        fw_res_out = norm1_out + fw_out
        norm2_out = self.dropout2(self.norm2(fw_res_out))

        return norm2_out
