import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttentionWithParaTopicNorm(nn.Module):

    def __init__(self, dropout, d_model, d_k, d_v, pos_win, device):
        super(ScaledDotProductAttentionWithParaTopicNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k
        self.pos_win = pos_win
        self.d_model = d_model
        self.device = device

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, pt_attn, bias):
        """
        :param q: [batch_size, n_heads, len_q, dim_per_head]
        :param k: [batch_size, n_heads, len_k_s, dim_per_head]
        :param v: [batch_size, n_heads, len_v_s, dim_per_head]
        :param bias: [batch_size, n_heads, len_q, len_k_s]
        :param pt_attn: [batch_size, n_heads, len_q, max_para_num]
        len_q = len_k = len_v = n_blocks
        d_k = d_v = dim_per_head
        """
        d_model = self.d_model
        batch_size, n_heads = q.size(0), q.size(1)
        len_q, len_k_s = q.size(2), k.size(2)
        scaled_q = q / (self.d_k ** 0.5)
        # [batch_size, n_heads, len_q, len_k_s]
        attn = torch.matmul(scaled_q, k.transpose(2, 3))
        if bias is not None:
            attn += bias

        # [batch_size, n_heads, len_q, len_k_s]
        weights = pt_attn * F.softmax(attn, dim=-1)
        weights = self.dropout(weights)

        # [batch_size, n_heads, len_q, dim_per_head]
        graph_out = torch.matmul(weights, v)
        # [batch_size, len_q, d_model]
        graph_out = graph_out.transpose(1, 2).contiguous().view(batch_size, len_q, d_model)
        graph_out = self.fc_out(graph_out)

        # [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k_s]
        return graph_out, weights
