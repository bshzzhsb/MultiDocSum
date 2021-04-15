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
        :param q: [batch_size, n_heads, len_q, dim_per_head]
        :param k: [batch_size, n_heads, len_k, dim_per_head]
        :param v: [batch_size, n_heads, len_v, dim_per_head]
        :param bias: [batch_size, n_heads, len_q, len_k]
        :return:
        """
        # [batch_size, n_heads, len_q, len_k]
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))

        if bias is not None:
            attn += bias

        # [batch_size, n_heads, len_q, len_k]
        weights = self.dropout(F.softmax(attn, dim=-1))
        # [batch_size, n_heads, len_q, dim_per_head]
        output = torch.matmul(weights, v)
        return output, weights


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
        weights = self.dropout(F.softmax(product, dim=-1))
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

        weights = self.dropout(F.softmax(attn, dim=-1))

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
        self.device = device

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
            pos_s = self.fc_pos_s(torch.tanh(pos_v))
            # pos 表示摘要 q 中的每个单词关注的段落，由于段落为整数，所以分为了 pos_up 和 pos_down
            pos = torch.sigmoid(pos_s) * (len_k_s - 1)

            # [batch_size, n_heads, len_q, 1]
            pos_up = torch.ceil(pos).to(torch.int64)
            pos_down = torch.floor(pos).to(torch.int64)

            batch_ind = torch.arange(0, batch_size, 1, dtype=torch.int64, device=self.device)
            # [batch_size, n_heads, len_q, 1]
            batch_ind = batch_ind.view(batch_size, 1, 1, 1).expand(-1, n_heads, len_q, -1)

            head_ind = torch.arange(0, n_heads, 1, dtype=torch.int64, device=self.device)
            # [batch_size, n_heads, len_q, 1]
            head_ind = head_ind.view(1, n_heads, 1, 1).expand(batch_size, -1, len_q, -1)

            query_ind = torch.arange(start=0, end=len_q, step=1, dtype=torch.int64, device=self.device)
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

            pos_up_ind = pos_up_ind.permute(3, 0, 1, 2)
            pos_down_ind = pos_down_ind.permute(3, 0, 1, 2)
            # 将 pos_up/pos_down 中对应的中心段落与其他段落的相似度，作为摘要与其他段落的相似度
            # [batch_size, n_heads, len_q, len_k_s]
            graph_attn_mask_up = graph_attn_mask[list(pos_up_ind)]
            graph_attn_mask_down = graph_attn_mask[list(pos_down_ind)]

            # 将 pos_up/pos_down 与 pos 的距离作为权重，加权得到最终的 graph_attn_mask
            # [batch_size, n_heads, len_q, len_k_s]
            graph_attn_mask_select = graph_attn_mask_up * (1.0 - (pos_up.to(torch.float32) - pos)) + \
                graph_attn_mask_down * (1.0 - (pos - pos_down.to(torch.float32)))

            gaussian_w = (-0.5 * graph_attn_mask_select * graph_attn_mask_select) / ((0.5 * self.pos_win) ** 2)

            attn += gaussian_w

        # [batch_size, n_heads, len_q, len_k_s]
        weights = self.dropout(F.softmax(attn, dim=-1))

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

        weights = F.softmax(attn, dim=-1)

        # [batch_size, n_heads, len_q, len_k, n_tokens]
        attn_w = weights.transpose(1, 2).transpose(2, 3)
        # [batch_size, n_heads, len_q, len_k, n_tokens]
        attn_w = torch.mul(attn_w, attn_s.unsqueeze(-1))
        # [batch_size, n_heads, len_q, len_k * n_tokens]
        attn_w = attn_w.contiguous().view(batch_size, n_heads, len_q, -1)

        attn_w = self.dropout(attn_w)

        # [batch_size, n_heads, len_v * n_tokens, d_v]
        v_w = v.transpose(1, 2).contiguous().view(batch_size, n_heads, -1, d_v)

        # [batch_size, n_heads, len_q, d_v]
        out = torch.matmul(attn_w, v_w)
        # [batch_size, len_q, n_heads * d_v]
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)

        out = self.fc(out)

        # [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k * n_tokens]
        return out, attn_w
