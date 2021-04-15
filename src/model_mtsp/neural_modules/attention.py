import torch
import torch.nn as nn

from models.neural_modules.attention_modules import \
    ScaledDotProductAttention, \
    ScaledDotProductAttentionWithSentenceNorm
from model_mtsp.neural_modules.attention_modules import ScaledDotProductAttentionWithParaTopic


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
        self.w_qs_t = nn.Linear(d_model, n_heads * d_k)
        self.w_ks_t = nn.Linear(d_model, n_heads * d_k)
        self.w_vs_t = nn.Linear(d_model, n_heads * d_v)
        self.fc_pt = nn.Linear(d_model, n_heads * d_k)
        self.fc_topic = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model * 3, d_model)

        self.graph_attn = ScaledDotProductAttentionWithParaTopic(dropout, d_model, d_k, d_v, pos_win, self.device)
        self.topic_attn = ScaledDotProductAttention(dropout, d_k)
        self.attn_with_sent_norm = ScaledDotProductAttentionWithSentenceNorm(
            dropout, d_model, d_k, d_v, n_heads
        )

    def forward(self, q, k_s, v_s, k_w, v_w, bias_w, bias_s, graph_attn_bias,
                topic=None, topic_attn_bias=None, pt_attn=None, cache=None):
        """
        :param q: [batch_size, seq_len, d_model]
        :param k_s: [batch_size, n_blocks, d_model]
        :param v_s: [batch_size, n_blocks, d_model]
        :param k_w: [batch_size, n_blocks, n_tokens, d_model]
        :param v_w: [batch_size, n_blocks, n_tokens, d_model]
        :param bias_w: [batch_size, n_blocks, n_heads, seq_len, n_tokens]
        :param bias_s: [batch_size, n_heads, seq_len, n_blocks]
        :param graph_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param topic: []
        :param topic_attn_bias: []
        :param pt_attn: [batch_size, n_blocks, d_model]
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

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * dim_per_head)

        pt_attn = shape(self.fc_pt(pt_attn))

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
        context_s, attn_s = self.graph_attn(q_s, k_s, v_s, pt_attn, bias_s)

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

        q_t = self.w_qs_t(q)
        q_t = shape(q_t)
        if cache is not None:
            if cache['memory_k_topic'] is not None:
                k_t, v_t = cache['memory_k_topic'], cache['memory_v_topic']
            else:
                k_t, v_t = self.w_ks_t(topic), self.w_vs_t(topic)
                k_t, v_t = shape(k_t), shape(v_t)
                cache['memory_k_topic'], cache['memory_v_topic'] = k_t, v_t
        else:
            # [batch_size, n_topic_words, n_heads * dim_per_head]
            k_t, v_t = self.w_ks_t(topic), self.w_vs_t(topic)
            # [batch_size, n_heads, n_topic_words, dim_per_head]
            k_t, v_t = shape(k_t), shape(v_t)
        # [batch_size, n_heads, len_q, dim_per_head]
        context_t, _ = self.topic_attn(q_t, k_t, v_t, topic_attn_bias)

        # [batch_size, len_q, d_model]
        context_t = unshape(context_t)
        context_t = self.fc_topic(context_t)

        # [batch_size, len_q, d_model * 3]
        out = torch.cat([context_t, context_s, context_w], dim=2)
        # [batch_size, len_q, d_model]
        out = self.fc(out)

        # [batch_size, len_q, d_model]
        return out
