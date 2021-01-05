import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout, d_k, bias):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k
        self.bias = bias

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e-9)

        if self.bias:
            attn += self.bias

        weights = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(weights, v)
        return output, weights


class DotProductPooling(nn.Module):

    def __init__(self, dropout, bias):
        super(DotProductPooling, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

    def forward(self, k, v):
        """
        TODO: bias
        :param k: [batch_size, n_heads, seq_len, 1]
        :param v: [batch_size, n_heads, seq_len, d_v]
        """
        k = k.squeeze(-1)
        weights = self.dropout(F.softmax(k))
        out = torch.mul(v, weights)  # [batch_size, n_heads, seq_len, d_v]
        out = out.sum(dim=2)  # [batch_size, n_heads, d_v]

        return out


class GraphScaledDotProductAttention(nn.Module):

    def __init__(self, dropout, d_k, bias, graph_attn_bias, pos_win):
        super(GraphScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k
        self.bias = bias
        self.graph_attn_bias = graph_attn_bias
        self.pos_win = pos_win

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))
        if self.bias:
            attn += self.bias

        if self.graph_attn_bias:
            gaussian_w = (-0.5 * (self.graph_attn_bias ** 2)) / ((0.5 * self.pos_win) ** 2)
            attn += gaussian_w

        weights = self.dropout(F.softmax(attn))

        graph_out = torch.matmul(weights, v)
        return graph_out, weights


class MultiHeadAttention(nn.Module):
    """
    TODO: mask
    多头注意力是通过 transpose 和 reshape 实现的，而不是真的拆分张量
    In practice, the multi-headed attention are done with transposes and reshapes
    rather than actual separate tensors.
    """
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=bias)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=bias)

        self.attn = ScaledDotProductAttention(dropout, d_k, bias)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        :param q: [batch_size, sequence_length, d_model]
        d_model = num_heads * d_k
        """
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_heads, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_heads, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_heads, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out, attn = self.attn(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        out = self.dropout(self.fc(out))
        out += residual
        out = self.layer_norm(out)

        return out, attn


class MultiHeadPooling(nn.Module):

    def __init__(self, n_heads, d_model, d_v, bias, dropout=0):
        super(MultiHeadPooling, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v

        self.w_ks = nn.Linear(d_model, n_heads, bias=bias)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=bias)
        self.fc = nn.Linear(d_model, d_model)

        self.attn = DotProductPooling(dropout, bias)

    def forward(self, k, v, mask=None):
        """
        :param k: [batch_size, sequence_length, d_model]
        :param v: same as k
        """
        batch_size, d_v, n_heads = k.size(0), self.d_v, self.n_heads
        k = self.w_ks(k).view(batch_size, -1, n_heads, 1)  # [batch_size, seq_len, n_heads, 1]
        v = self.w_vs(v).view(batch_size, -1, n_heads, d_v)  # [batch_size, seq_len, n_heads, d_v]

        k, v = k.transpose(1, 2), v.transpose(1, 2)

        out = self.attn(k, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v).squeeze(1)
        out = self.fc(out)

        return out  # [batch_size, d_model]


class MultiHeadStructureAttention(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, bias, graph_attn_bias, pos_win, dropout=0):
        super(MultiHeadStructureAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=bias)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=bias)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.graph_attn = GraphScaledDotProductAttention(dropout, d_k, bias, graph_attn_bias, pos_win)

    def forward(self, q, k, v):
        """
        :param q: [batch_size, sequence_length, d_model]
        d_model = num_heads * d_k
        """
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_heads, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_heads, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_heads, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out, _ = self.graph_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        out = self.dropout(self.fc(out))
        out += residual
        out = self.layer_norm(out)

        return out


class MultiHeadHierarchicalAttention(nn.Module):

    def __init__(self, bias_w, bias_s, graph_attn_bias, pos_win, d_k, d_v, d_model, n_heads=1, dropout=0):
        super(MultiHeadHierarchicalAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.fc1 = nn.Linear(d_model, d_model)

        self.graph_attn = GraphScaledDotProductAttention(dropout, d_k, bias_s, graph_attn_bias, pos_win)

    def forward(self, q, k_w, v_w, k_s, v_s):
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k_s, len_v_s = q.size(0), q.size(1), k_s.size(1), v_s.size(1)

        q_s = self.w_qs(q).view(batch_size, len_q, n_heads, d_k)
        k_s = self.w_ks(k_s).view(batch_size, len_k_s, n_heads, d_k)
        v_s = self.w_vs(v_s).view(batch_size, len_v_s, n_heads, d_v)

        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)

        context_s, attns = self.graph_attn(q_s, k_s, v_s)
        context_s = context_s.transpose(1, 2).continugous().view(batch_size, len_q, -1)
        context_s = self.fc1(context_s)
