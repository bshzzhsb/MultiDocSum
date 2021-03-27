import torch.nn as nn

from models.neural_modules.attention import MultiHeadAttention, MultiHeadHierarchicalAttention, MultiHeadTopicAttention
from models.neural_modules.neural_modules import PositionwiseFeedForward
from models.layers.encoder import SelfAttentionPoolingLayer


class TransformerDecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.context_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)

        self.pos_ffd = PositionwiseFeedForward(d_model, d_inner_hidden, dropout)

    def forward(self, dec_input, enc_output, self_attn_bias, dec_enc_attn_bias):
        q = self.layer_norm_1(dec_input)
        self_attn_output = self.self_attn(q, q, q, self_attn_bias)

        self_attn_output = self.dropout(self_attn_output) + dec_input

        q = self.layer_norm_2(self_attn_output)
        context_attn_output = self.context_attn(q, enc_output, enc_output, dec_enc_attn_bias)

        context_attn_output = self.dropout(context_attn_output) + self_attn_output

        dec_output = self.pos_wise_ffd(context_attn_output)

        return dec_output


class TransformerDecoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner_hidden, dropout):
        super(TransformerDecoder, self).__init__()
        self.n_layers = n_layers

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.transformer_decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden, dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, dec_input, enc_output, dec_self_attn_bias, dec_enc_attn_bias):
        for i in range(self.n_layers):
            dec_output = self.transformer_decoder_layers[i](
                dec_input, enc_output, dec_self_attn_bias, dec_enc_attn_bias)
            dec_input = dec_output

        dec_output = self.layer_norm(dec_output)

        return dec_output


class GraphDecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win, dropout, device, topic=None):
        super(GraphDecoderLayer, self).__init__()
        self.topic = topic

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.multi_head_hierarchical_attn = MultiHeadHierarchicalAttention(
            pos_win, d_k, d_v, d_model, device, n_heads, dropout, topic=topic
        )

        self.pos_ffd = PositionwiseFeedForward(d_model, d_inner_hidden, dropout)

    def forward(self, dec_input, enc_words_output, enc_sents_output, self_attn_bias,
                dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                cache=None):
        # [batch_size, tgt_len, d_model]
        q = self.layer_norm_1(dec_input)
        # [batch_size, tgt_len, d_model]
        self_attn_output = self.self_attn(q, q, q, self_attn_bias, cache=cache, type='self')
        self_attn_output = self.dropout(self_attn_output) + dec_input

        q = self.layer_norm_2(self_attn_output)
        # [batch_size, tgt_len, d_model]
        hier_attn_output = self.multi_head_hierarchical_attn(
            q, enc_sents_output, enc_sents_output, enc_words_output, enc_words_output,
            dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
            cache=cache
        )
        hier_attn_output = self.dropout(hier_attn_output) + self_attn_output

        # [batch_size, len_q, d_model]
        dec_output = self.pos_ffd(hier_attn_output)

        # [batch_size, len_q, d_model]
        return dec_output


class GraphDecoder(nn.Module):

    def __init__(self, batch_size, n_layers, n_heads, d_model, d_k, d_v,
                 d_inner_hidden, pos_win, dropout, device, topic=None):
        super(GraphDecoder, self).__init__()
        self.n_layers = n_layers
        self.topic = topic

        if self.topic == 'sum':
            self.pooling = SelfAttentionPoolingLayer(n_heads, d_model, d_v, batch_size, dropout)
            self.topic_attn = MultiHeadTopicAttention(n_heads, d_model, dropout)

        self.graph_decoder_layers = nn.ModuleList([
            GraphDecoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win, dropout, device
            ) for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_words_output, enc_sents_output, dec_self_attn_bias,
                dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                topic=None, topic_bias=None, state=None):
        if self.topic == 'sum':
            # [batch_size * tgt_len, n_topic_words, d_model]
            dec_input = self.topic_attn(topic, dec_input, dec_input, topic_bias)
            # [batch_size, tgt_len, d_model]
            dec_input = self.pooling(dec_input)

        for i in range(self.n_layers):
            # [batch_size, len_q, d_model]
            dec_output = self.graph_decoder_layers[i](
                dec_input, enc_words_output, enc_sents_output,
                dec_self_attn_bias, dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                cache=state.cache['layer_{}'.format(i)] if state is not None and state.cache is not None else None
            )
            dec_input = dec_output

        dec_output = self.layer_norm(dec_output)

        # [batch_size, len_q, d_model]
        return dec_output

    def init_decoder_state(self, with_cache=False):
        state = GraphDecoderState()
        if with_cache:
            state.init_cache(self.n_layers)
        return state


class GraphDecoderState(object):

    def __init__(self):
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    def init_cache(self, num_layers):
        self.cache = {}
        for layer in range(num_layers):
            layer_cache = {
                'memory_keys': None,
                'memory_values': None,
                'self_keys': None,
                'self_values': None,
                'static_k_sent': None,
                'static_v_sent': None,
                'static_k_word': None,
                'static_v_word': None
            }
            self.cache['layer_{}'.format(layer)] = layer_cache

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.cache is not None:
            _recursive_map(self.cache)
