import torch.nn as nn

from models.neural_modules.attention import MultiHeadAttention
from model_topic_kvs.neural_modules.attention import MultiHeadHierarchicalAttention
from models.neural_modules.neural_modules import PositionwiseFeedForward


class GraphDecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win, dropout, device, topic=None):
        super(GraphDecoderLayer, self).__init__()
        self.topic = topic
        self.n_heads = n_heads

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.multi_head_hierarchical_attn = MultiHeadHierarchicalAttention(
            pos_win, d_k, d_v, d_model, device, n_heads, dropout, topic=topic
        )

        self.pos_ffd = PositionwiseFeedForward(d_model, d_inner_hidden, dropout)

    def forward(self, dec_input, enc_words_output, enc_sents_output, self_attn_bias,
                dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                topic_embed_out, tgt_topic_attn_bias, cache=None):
        # [batch_size, tgt_len, d_model]
        q = self.layer_norm_1(dec_input)
        # [batch_size, tgt_len, d_model]
        self_attn_output = self.self_attn(q, q, q, self_attn_bias, cache=cache, type='self')
        self_attn_output = self.dropout1(self_attn_output) + dec_input

        q = self.layer_norm_2(self_attn_output)
        # [batch_size, tgt_len, d_model]
        hier_attn_output = self.multi_head_hierarchical_attn(
            q, enc_sents_output, enc_sents_output, enc_words_output, enc_words_output,
            dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
            topic_embed_out, tgt_topic_attn_bias, cache=cache
        )
        hier_attn_output = self.dropout2(hier_attn_output) + self_attn_output

        # [batch_size, len_q, d_model]
        dec_output = self.pos_ffd(hier_attn_output)

        # [batch_size, len_q, d_model]
        return dec_output


class GraphDecoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner_hidden,
                 pos_win, dropout, device):
        super(GraphDecoder, self).__init__()
        self.n_layers = n_layers

        self.graph_decoder_layers = nn.ModuleList([
            GraphDecoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win, dropout, device
            ) for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_words_output, enc_sents_output, dec_self_attn_bias,
                dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                topic_embed_out, tgt_topic_attn_bias, state=None):
        for i in range(self.n_layers):
            # [batch_size, len_q, d_model]
            dec_output = self.graph_decoder_layers[i](
                dec_input, enc_words_output, enc_sents_output, dec_self_attn_bias,
                dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                topic_embed_out, tgt_topic_attn_bias,
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
                'static_v_word': None,
                'memory_k_topic': None,
                'memory_v_topic': None
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
