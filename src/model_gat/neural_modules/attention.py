import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):

    def __init__(self, n_heads, in_features, out_features, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.n_heads = n_heads
        self.out_features = out_features

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features // n_heads, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, h, mask):
        """
        :param h: [batch_size, n_blocks, d_model]
        :param mask: [batch_size, n_heads, n_blocks, n_blocks]
        :return: [batch_size, n_blocks, d_model]
        """
        n_blocks, d_model, n_heads = h.size(1), self.out_features, self.n_heads
        dim_per_head = d_model // n_heads

        # [batch_size, n_blocks, d_model]
        h = self.fc(h)
        # [batch_size, n_heads, n_blocks, dim_per_head]
        h = h.view(-1, n_blocks, n_heads, dim_per_head).transpose(1, 2)

        # [batch_size, n_heads, n_blocks * n_blocks, dim_per_head]
        h_repeat_in_chunks = h.repeat_interleave(n_blocks, dim=2)
        # [batch_size, n_heads, n_blocks * n_blocks, dim_per_head]
        h_repeat_alternating = h.repeat(1, 1, n_blocks, 1)

        # [batch_size, n_heads, n_blocks * n_blocks, dim_per_head * 2]
        attn_input = torch.cat([h_repeat_in_chunks, h_repeat_alternating], dim=-1)
        # [batch_size, n_heads, n_blocks, n_blocks, dim_per_head * 2]
        attn_input = attn_input.view(-1, n_heads, n_blocks, n_blocks, 2 * dim_per_head)

        # [batch_size, n_heads, n_blocks, n_blocks]
        e = self.leaky_relu(self.a(attn_input).squeeze(-1))

        # zero_vec = -9e15 * torch.ones_like(e)
        # attn = torch.where(adj > 0, e, zero_vec)
        attn = e + mask[:, 0: n_heads]

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # [batch_size, n_heads, n_blocks, dim_per_head]
        h_prime = torch.matmul(attn, h)

        h_prime = F.elu(h_prime)
        # [batch_size, n_blocks, d_model]
        h_prime = h_prime.transpose(1, 2).contiguous().view(-1, n_blocks, d_model)

        return h_prime
