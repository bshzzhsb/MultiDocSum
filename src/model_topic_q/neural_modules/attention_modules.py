import torch
import torch.nn as nn
import torch.nn.functional as F


class TopicScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, dropout):
        super(TopicScaledDotProductAttention, self).__init__()
        self.d_k = d_k

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, bias):
        """
        q: topic matrix; k, v: summary matrix
        :param q: [batch_size, n_heads, n_topic_words, dim_per_head]
        :param k: [batch_size, n_heads, tgt_len, dim_per_head]
        :param v: [batch_size, n_heads, tgt_len, dim_per_head]
        :param bias: [batch_size, tgt_len, n_heads, n_topic_words, tgt_len]
        :return: [batch_size, n_topic_words * tgt_len, d_model]
        """
        batch_size, n_heads, tgt_len, dim_per_head = k.size(0), k.size(1), k.size(2), self.d_k
        # [batch_size, tgt_len, n_heads, n_topic_words, dim_per_head]
        q = q.unsqueeze(1).expand(-1, tgt_len, -1, -1, -1)

        # [batch_size, tgt_len, n_heads, n_topic_words, tgt_len]
        weight = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3).unsqueeze(1))

        if bias is not None:
            weight += bias

        # [batch_size, tgt_len, n_heads, n_topic_words, tgt_len]
        weight = F.softmax(self.dropout(weight), dim=-1)
        # [batch_size, n_heads, n_topic_words * tgt_len, tgt_len]
        weight = weight.transpose(1, 2).contiguous().view(batch_size, n_heads, -1, tgt_len)

        # [batch_size, n_heads, n_topic_words * tgt_len, dim_per_head]
        attn = torch.matmul(weight, v)
        # [batch_size, n_topic_words * tgt_len, d_model]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, dim_per_head * n_heads)

        return attn
