import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural_modules.attention_modules import \
    ScaledDotProductAttention, DotProductPooling, \
    GraphScaledDotProductAttention, \
    GraphScaledDotProductAttentionWithMask, \
    ScaledDotProductAttentionWithSentenceNorm \


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

    def forward(self, q, k, v, bias, cache=None, type=None):
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

        dim_per_head, n_heads = self.d_k, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        def shape(x):
            # [batch_size, n_heads, len, dim_per_head]
            return x.view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * dim_per_head)

        if cache is not None:
            if type == 'self':
                q, k, v = self.w_qs(q), self.w_ks(k), self.w_vs(v)
                k, v = shape(k), shape(v)

                device = k.device
                if cache['self_keys'] is not None:
                    k = torch.cat([cache['self_keys'].to(device), k], dim=2)
                if cache['self_values'] is not None:
                    v = torch.cat([cache['self_values'].to(device), v], dim=2)
                cache['self_keys'] = k
                cache['self_values'] = v
            elif type == 'context':
                q = self.w_qs(q)
                if cache['memory_keys'] is None:
                    k, v = self.w_ks(k), self.w_vs(v)
                    k = shape(k)
                    v = shape(v)
                else:
                    k, v = cache['memory_keys'], cache['memory_values']
                cache['memory_keys'] = k
                cache['memory_values'] = v
        else:
            q, k, v = self.w_qs(q), self.w_ks(k), self.w_vs(v)
            k, v = shape(k), shape(v)

        q = shape(q)

        # [batch_size, n_heads, len_q, d_v]
        out, _ = self.attn(q, k, v, bias)

        # [batch_size, len_q, d_model]
        out = unshape(out)
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
        out = out.transpose(1, 2).contiguous().view(batch_size, n_heads * d_v)
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

    def __init__(self, pos_win, d_k, d_v, d_model, device, n_heads=1, dropout=0, topic=None):
        super(MultiHeadHierarchicalAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.topic = topic

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

    def forward(self, q, k_s, v_s, k_w, v_w, bias_w, bias_s, graph_attn_bias, cache=None):
        """
        :param q: [batch_size, seq_len, d_model]
        :param k_s: [batch_size, n_blocks, d_model]
        :param v_s: [batch_size, n_blocks, d_model]
        :param k_w: [batch_size, n_blocks, n_tokens, d_model]
        :param v_w: [batch_size, n_blocks, n_tokens, d_model]
        :param bias_w: [batch_size, n_blocks, n_heads, seq_len, n_tokens]
        :param bias_s: [batch_size, n_heads, seq_len, n_blocks]
        :param graph_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param cache:
        :return: [batch_size, len_q, d_model]
        d_model = dim_embed = n_heads * d_k = n_heads * d_v
        len_k_s = len_v_s = len_k_w = len_v_w = n_blocks
        """
        dim_per_head, n_heads = self.d_k, self.n_heads
        batch_size, len_q = q.size(0), q.size(1)
        len_k_s, len_v_s, len_k_w, len_v_w = \
            k_s.size(1), v_s.size(1), k_w.size(1), v_w.size(1)

        def shape(x):
            return x.view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)

        def shape_w(x):
            return x.view(batch_size, len_k_w, -1, n_heads, dim_per_head).transpose(2, 3)

        q_s = self.w_qs_s(q)
        q_s = shape(q_s)

        if cache is not None:
            if cache['static_k_sent'] is not None:
                k_s, v_s = cache['static_k_sent'], cache['static_v_sent']
            else:
                k_s, v_s = self.w_ks_s(k_s), self.w_vs_s(v_s)
                k_s, v_s = shape(k_s), shape(v_s)
                cache['static_k_sent'], cache['static_v_sent'] = k_s, v_s
        else:
            k_s, v_s = self.w_ks_s(k_s), self.w_vs_s(v_s)
            k_s, v_s = shape(k_s), shape(v_s)

        # [batch_size, len_q, d_model] [batch_size, n_heads, len_q, len_k_s]
        context_s, attn_s = self.graph_attn(q_s, k_s, v_s, bias_s, graph_attn_bias)

        q_w = self.w_qs_w(q)
        # [batch_size, n_heads, len_q, d_k]
        q_w = shape(q_w)

        if cache is not None:
            if cache['static_k_word'] is not None:
                k_w, v_w = cache['static_k_word'], cache['static_v_word']
            else:
                k_w, v_w = self.w_ks_w(k_w), self.w_vs_w(v_w)
                k_w, v_w = shape_w(k_w), shape_w(v_w)
                cache['static_k_word'], cache['static_v_word'] = k_w, v_w
        else:
            k_w, v_w = self.w_ks_w(k_w), self.w_vs_w(v_w)
            # [batch_size, len_k_w, n_heads, n_tokens, d_k]
            # [batch_size, len_v_w, n_heads, n_tokens, d_v]
            k_w, v_w = shape_w(k_w), shape_w(v_w)

        # [batch_size, len_q, d_model]
        context_w, _ = self.attn_with_sent_norm(q_w, k_w, v_w, attn_s, bias_w)

        # [batch_size, len_q, d_model * 2]
        out = torch.cat([context_s, context_w], dim=2)
        # [batch_size, len_q, d_model]
        out = self.fc(out)

        # [batch_size, len_q, d_model]
        return out
