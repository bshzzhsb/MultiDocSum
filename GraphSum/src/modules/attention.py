import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k

    def forward(self, q, k, v, bias=None):
        """
        :param q: [batch_size, n_heads, len_q, d_k]
        :param k: [batch_size, n_heads, len_k, d_k]
        :param v: [batch_size, n_heads, len_v, d_v]
        :param bias: [batch_size, n_heads, len_q, len_k]
        len_q = len_k = len_v
        d_model = n_heads * d_k = n_heads * d_v
        """
        # [batch_size, n_heads, len_q, len_k]
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))

        if bias is not None:
            attn += bias

        # [batch_size, n_heads, len_q, len_k]
        weights = self.dropout(F.softmax(attn, dim=-1))
        # [batch_size, n_heads, len_q, d_v]
        output = torch.matmul(weights, v)
        return output


class DotProductPooling(nn.Module):

    def __init__(self, dropout):
        super(DotProductPooling, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, bias):
        """
        :param k: [batch_size, n_heads, len_k, 1]
        :param v: [batch_size, n_heads, len_v, d_v]
        :param bias: [batch_size, n_heads, len_k, len_k]
        len_k = len_v
        """
        # [batch_size, n_heads, len_k]
        product = k.squeeze(-1)
        if bias is not None:
            # [batch_size, n_heads, 1, len_k]
            bias_sliced = bias[:, :, 0, :]
            product += bias_sliced.squeeze(2)

        # [batch_size, n_heads, len_k]
        weights = self.dropout(F.softmax(product))
        # [batch_size, n_heads, len_k, d_v]
        out = v * weights.unsqueeze(-1)
        # [batch_size, n_heads, d_v]
        out = out.sum(dim=2)

        return out


class GraphScaledDotProductAttention(nn.Module):

    def __init__(self, dropout, d_k, pos_win):
        super(GraphScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k
        self.pos_win = pos_win

    def forward(self, q, k, v, bias, graph_attn_bias):
        """
        :param q: [batch_size, n_heads, len_q, d_k]
        :param k: [batch_size, n_heads, len_k, d_k]
        :param v: [batch_size, n_heads, len_v, d_v]
        :param bias: [batch_size, n_heads, len_q, len_k]
        :param graph_attn_bias: [batch_size, n_heads, len_q, len_k]
        len_q = len_k = len_v = n_blocks
        d_k = d_v = dim_per_head
        """
        scaled_q = q / (self.d_k ** 0.5)
        # [batch_size, n_heads, len_q, len_k]
        attn = torch.matmul(scaled_q, k.transpose(2, 3))
        if bias is not None:
            attn += bias

        if graph_attn_bias is not None:
            # [batch_size, n_heads, len_q, len_k]
            gaussian_w = (-0.5 * (graph_attn_bias * graph_attn_bias)) / ((0.5 * self.pos_win) ** 2)
            attn += gaussian_w

        weights = self.dropout(F.softmax(attn))

        # [batch_size, n_heads, len_q, d_v]
        graph_out = torch.matmul(weights, v)
        return graph_out, weights


class GraphScaledDotProductAttentionWithMask(nn.Module):

    def __init__(self, dropout, d_model, d_k, d_v, pos_win, device):
        super(GraphScaledDotProductAttentionWithMask, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k
        self.pos_win = pos_win
        self.d_model = d_model

        self.fc_pos_v = nn.Linear(d_k, d_v)
        self.fc_pos_s = nn.Linear(d_v, 1)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, bias, graph_attn_bias):
        """
        :param q: [batch_size, n_heads, len_q, dim_per_head]
        :param k: [batch_size, n_heads, len_k_s, dim_per_head]
        :param v: [batch_size, n_heads, len_v_s, dim_per_head]
        :param bias: [batch_size, n_heads, len_q, len_k_s]
        :param graph_attn_bias: [batch_size, n_heads, len_k_s, len_k_s]
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

        if graph_attn_bias is not None:
            # [batch_size, n_heads, len_q, d_v]
            pos_v = self.fc_pos_v(scaled_q)
            # [batch_size, n_heads, len_q, 1]
            pos_s = self.fc_pos_s(F.tanh(pos_v))
            pos = F.sigmoid(pos_s) * (len_k_s - 1)

            # [batch_size, n_heads, len_q, 1]
            pos_up = torch.ceil(pos).to(torch.int64)
            pos_down = torch.floor(pos).to(torch.int64)

            batch_ind = torch.arange(0, batch_size, 1, dtype=torch.int64)
            # [batch_size, n_heads, len_q, 1]
            batch_ind = batch_ind.view(batch_size, 1, 1, 1).expand(-1, n_heads, len_q, -1)

            head_ind = torch.arange(0, n_heads, 1, dtype=torch.int64)
            # [batch_size, n_heads, len_q, 1]
            head_ind = head_ind.view(1, n_heads, 1, 1).expand(batch_size, -1, len_q, -1)

            query_ind = torch.arange(start=0, end=len_q, step=1, dtype=torch.int64)
            # [batch_size, n_heads, len_q, 1]
            query_ind = query_ind.view(1, 1, len_q, 1).expand(batch_size, n_heads, -1, -1)

            # [batch_size, n_heads, len_q, 4]
            pos_up_ind = torch.cat((batch_ind, head_ind, query_ind, pos_up), dim=3).to(self.device)
            pos_up_ind.requires_grad_(requires_grad=False)
            pos_down_ind = torch.cat((batch_ind, head_ind, query_ind, pos_down), dim=3).to(self.device)
            pos_down_ind.requires_grad_(requires_grad=False)

            # [batch_size, n_heads, len_q, len_k_s, len_k_s]
            graph_attn_mask = torch.unsqueeze(graph_attn_bias, dim=2)
            graph_attn_mask = graph_attn_mask.expand(-1, -1, len_q, -1, -1)

            # [batch_size, n_heads, len_q, len_k_s]
            pos_up_ind = pos_up_ind.transpose(2, 3).transpose(1, 2).transpose(0, 1)
            graph_attn_mask_up = graph_attn_mask[pos_up_ind.numpy().tolist()]
            pos_down_ind = pos_down_ind.transpose(2, 3).transpose(1, 2).transpose(0, 1)
            graph_attn_mask_down = graph_attn_mask[pos_down_ind.numpy().tolist()]

            # [batch_size, n_heads, len_q, len_k_s]
            graph_attn_mask_select = graph_attn_mask_up * (1.0 - (pos_up.to(torch.float32) - pos)) + \
                graph_attn_mask_down * (1.0 - (pos - pos_down.to(torch.float32)))

            gaussian_w = (-0.5 * graph_attn_mask_select * graph_attn_mask_select) / ((0.5 * self.pos_win) ** 2)

            attn += gaussian_w

        # [batch_size, n_heads, len_q, len_k_s]
        weights = self.dropout(F.softmax(attn))

        # [batch_size, n_heads, len_q, dim_per_head]
        graph_out = torch.matmul(weights, v)
        # [batch_size, len_q, d_model]
        graph_out = graph_out.transpose(1, 2).contiguous().view(batch_size, len_q, d_model)
        graph_out = self.fc_out(graph_out)

        # [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k_s]
        return graph_out, weights


class ScaledDotProductAttentionWithSentenceNorm(nn.Module):

    def __init__(self, dropout, d_model, d_k, d_v, n_heads):
        super(ScaledDotProductAttentionWithSentenceNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_k
        self.n_heads = n_heads
        self.d_v = d_v

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_s, bias):
        """
        :param q: [batch_size, n_heads, seq_len, dim_per_head]
        :param k: [batch_size, n_blocks, n_heads, n_tokens, dim_per_head]
        :param v: [batch_size, n_blocks, n_heads, n_tokens, dim_per_head]
        :param attn_s: [batch_size, n_heads, len_q, n_blocks]
        :param bias: [batch_size, n_blocks, n_heads, len_q, n_tokens]
        """
        batch_size, len_q, len_k = q.size(0), q.size(2), k.size(1)
        d_v, n_heads = self.d_v, self.n_heads

        # [batch_size, len_k, n_heads, len_q, d_k]
        q = q.unsqueeze(1).expand(-1, len_k, -1, -1, -1)
        # [batch_size, len_k, n_heads, len_q, n_tokens]
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(3, 4))

        if bias is not None:
            attn += bias

        weights = F.softmax(attn)

        # [batch_size, n_heads, len_q, len_k, n_tokens]
        attn_w = weights.transpose(1, 2).transpose(2, 3)
        # [batch_size, n_heads, len_q, len_k, n_tokens]
        attn_w = torch.mul(attn_w, attn_s.unsqueeze(-1))
        # [batch_size, n_heads, len_q, len_k * n_tokens]
        attn_w = attn_w.contiguous().view(batch_size, n_heads, len_q, -1)

        attn_w = self.dropout(attn_w)

        # [batch_size, n_heads, len_v * n_tokens, d_v]
        v_w = v.transpose(1, 2).view(batch_size, n_heads, -1, d_v)

        # [batch_size, n_heads, len_q, d_v]
        out = torch.matmul(attn_w, v_w)
        # [batch_size, len_q, n_heads * d_v]
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)

        out = self.fc(out)

        # [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k * n_tokens]
        return out, attn_w


class MultiHeadAttention(nn.Module):
    """
    多头注意力是通过 transpose 和 reshape 实现的，而不是真的拆分张量
    In practice, the multi-headed attention are done with transposes and reshapes
    rather than actual separate tensors.
    """
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout, d_k)

    def forward(self, q, k, v, bias):
        """
        :param q: [batch_size, seq_len, d_model]
        :param k: [batch_size, len_k, d_model]
        :param v: [batch_size, len_v, d_model]
        :param bias: [batch_size, n_heads, len_q, len_k]
        :return: [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k]
        d_model = num_heads * d_k
        """
        k = q if k is None else k
        v = k if v is None else v

        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # [batch_size, len, n_heads, dim_per_head]
        q = self.w_qs(q).view(batch_size, len_q, n_heads, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_heads, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_heads, d_v)

        # [batch_size, n_heads, len, dim_per_head]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # [batch_size, n_heads, len_q, d_v] [batch_size, n_heads, len_q, len_k]
        out = self.attn(q, k, v, bias)

        # [batch_size, len_q, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        out = self.fc(out)

        # [batch_size, len_q, d_model]
        return out


class MultiHeadPooling(nn.Module):

    def __init__(self, n_heads, d_model, d_v, dropout=0):
        super(MultiHeadPooling, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v

        self.w_ks = nn.Linear(d_model, n_heads)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(d_model, d_model)

        self.attn = DotProductPooling(dropout)

    def forward(self, k, v, bias):
        """
        :param k: [batch_size, len_k, d_model]
        :param v: [batch_size, len_v, d_model]
        :param bias: [batch_size, n_heads, len_k ,len_k]
        """
        batch_size, d_v, n_heads = k.size(0), self.d_v, self.n_heads
        # [batch_size, len_k, n_heads, 1]
        k = self.w_ks(k).view(batch_size, -1, n_heads, 1)
        # [batch_size, len_v, n_heads, d_v]
        v = self.w_vs(v).view(batch_size, -1, n_heads, d_v)

        # [batch_size, n_heads, len_k, 1] [batch_size, n_heads, len_v, d_v]
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        # [batch_size, n_heads, d_v]
        out = self.attn(k, v, bias)

        # [batch_size, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v).squeeze(1)
        out = self.fc(out)

        # [batch_size, d_model]
        return out


class MultiHeadStructureAttention(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, pos_win, dropout=0):
        super(MultiHeadStructureAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.graph_attn = GraphScaledDotProductAttention(dropout, d_k, pos_win)

    def forward(self, q, k, v, bias, graph_attn_bias):
        """
        :param q: [batch_size, len_q, d_model]
        :param k: [batch_size, len_k, d_model]
        :param v: [batch_size, len_v, d_model]
        :param bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param graph_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        d_model = num_heads * d_k = n_heads * d_v
        len_q = len_k = len_v = n_blocks
        """
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_heads, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_heads, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_heads, d_v)

        # [batch_size, n_heads, n_blocks, dim_per_head]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # [batch_size, n_heads, len_q, d_v]
        out, _ = self.graph_attn(q, k, v, bias, graph_attn_bias)
        # [batch_size, len_q, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        out = self.dropout(self.fc(out))
        out += residual
        out = self.layer_norm(out)

        # [batch_size, len_q, d_model]
        return out


class MultiHeadHierarchicalAttention(nn.Module):

    def __init__(self, pos_win, d_k, d_v, d_model, device, n_heads=1, dropout=0):
        super(MultiHeadHierarchicalAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device

        self.w_qs_s = nn.Linear(d_model, n_heads * d_k)
        self.w_ks_s = nn.Linear(d_model, n_heads * d_k)
        self.w_vs_s = nn.Linear(d_model, n_heads * d_v)
        self.w_qs_w = nn.Linear(d_model, n_heads * d_k)
        self.w_ks_w = nn.Linear(d_model, n_heads * d_k)
        self.w_vs_w = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(2 * d_model, d_model)

        self.graph_attn = GraphScaledDotProductAttentionWithMask(dropout, d_model, d_k, d_v, pos_win, self.device)
        self.attn_with_sent_norm = ScaledDotProductAttentionWithSentenceNorm(
            dropout, d_model, d_k, d_v, n_heads
        )

    def forward(self, q, k_s, v_s, k_w, v_w, bias_w, bias_s, graph_attn_bias):
        """
        :param q: [batch_size, seq_len, d_model]
        :param k_s: [batch_size, n_blocks, d_model]
        :param v_s: [batch_size, n_blocks, d_model]
        :param k_w: [batch_size, n_blocks, n_tokens, d_model]
        :param v_w: [batch_size, n_blocks, n_tokens, d_model]
        :param bias_w: [batch_size, n_blocks, n_heads, seq_len, n_tokens]
        :param bias_s: [batch_size, n_heads, seq_len, n_blocks]
        :param graph_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        d_model = dim_embed = n_heads * d_k = n_heads * d_v
        len_k_s = len_v_s = len_k_w = len_v_w = n_blocks
        """
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q = q.size(0), q.size(1)
        len_k_s, len_v_s, len_k_w, len_v_w, n_tokens = \
            k_s.size(1), v_s.size(1), k_w.size(1), v_w.size(1), k_w.size(2)

        q_s = self.w_qs_s(q).view(batch_size, len_q, n_heads, d_k)
        k_s = self.w_ks_s(k_s).view(batch_size, len_k_s, n_heads, d_k)
        v_s = self.w_vs_s(v_s).view(batch_size, len_v_s, n_heads, d_v)

        # [batch_size, n_heads, len, dim_per_head]
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)

        # [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k_s]
        context_s, attns = self.graph_attn(q_s, k_s, v_s, bias_s, graph_attn_bias)

        q_w = self.w_qs_w(q).view(batch_size, len_q, n_heads, d_k)
        k_w = self.w_ks_w(k_w).view(batch_size, len_k_w, n_tokens, n_heads, d_k)
        v_w = self.w_vs_w(v_w).view(batch_size, len_v_w, n_tokens, n_heads, d_v)

        # [batch_size, n_heads, len_q, d_k]
        # [batch_size, len_k_w, n_heads, n_tokens, d_k]
        # [batch_size, len_v_w, n_heads, n_tokens, d_v]
        q_w, k_w, v_w = q_w.transpose(1, 2), k_w.transpose(2, 3), v_w.transpose(2, 3)

        # [batch_size, len_q, d_model]
        context_w, _ = self.attn_with_sent_norm(q_w, k_w, v_w, attns, bias_w)

        # [batch_size, len_q, d_model * 2]
        out = torch.cat((context_s, context_w), dim=2)
        # [batch_size, len_q, d_model]
        out = self.fc(out)

        # [batch_size, len_q, d_model]
        return out
