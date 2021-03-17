import torch.nn as nn

from graph_sum.neural_modules.attention import MultiHeadAttention, MultiHeadHierarchicalAttention
from graph_sum.neural_modules.neural_modules import PositionWiseFeedForward, PreProcessLayer, PostProcessLayer


class TransformerDecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd, post_process_cmd):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn1 = MultiHeadAttention(n_heads, d_model, d_k, d_v, attn_dropout)
        self.self_attn2 = MultiHeadAttention(n_heads, d_model, d_k, d_v, attn_dropout)

        self.pre_process_layer1 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.pre_process_layer2 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.pre_process_layer3 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)

        self.post_process_layer1 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)
        self.post_process_layer2 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)
        self.post_process_layer3 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)

        self.pos_wise_ffd = PositionWiseFeedForward(d_model, d_inner_hidden, d_model, relu_dropout, hidden_act)

    def forward(self, dec_input, enc_output, self_attn_bias, dec_enc_attn_bias):
        q = self.pre_process_layer1(dec_input)
        self_attn_output = self.self_attn1(q, q, q, self_attn_bias)

        self_attn_output = self.post_process_layer1(dec_input, self_attn_output)

        q = self.pre_process_layer2(self_attn_output)
        context_attn_output = self.self_attn2(q, enc_output, enc_output, dec_enc_attn_bias)

        context_attn_output = self.post_process_layer2(self_attn_output, context_attn_output)

        x = self.pre_process_layer3(context_attn_output)
        ffd_output = self.pos_wise_ffd(x)

        dec_output = self.post_process_layer3(context_attn_output, ffd_output)

        return dec_output


class TransformerDecoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner_hidden,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd, post_process_cmd):
        super(TransformerDecoder, self).__init__()
        self.n_layers = n_layers

        self.transformer_decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden,
                pre_post_process_dropout, attn_dropout, relu_dropout,
                hidden_act, pre_process_cmd, post_process_cmd
            ) for i in range(n_layers)
        ])
        self.pre_process_layer = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)

    def forward(self, dec_input, enc_output, dec_self_attn_bias, dec_enc_attn_bias):
        for i in range(self.n_layers):
            dec_output = self.transformer_decoder_layers[i](
                dec_input, enc_output, dec_self_attn_bias, dec_enc_attn_bias)
            dec_input = dec_output

        dec_output = self.pre_process_layer(dec_output)

        return dec_output


class GraphDecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd, post_process_cmd, device):
        super(GraphDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, attn_dropout)
        self.multi_head_hierarchical_attn = MultiHeadHierarchicalAttention(
            pos_win=pos_win,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            device=device,
            n_heads=n_heads,
            dropout=attn_dropout
        )

        self.pre_process_layer1 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.pre_process_layer2 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.pre_process_layer3 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.post_process_layer1 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)
        self.post_process_layer2 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)
        self.post_process_layer3 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)

        self.pos_wise_ffd = PositionWiseFeedForward(d_model, d_inner_hidden, d_model, relu_dropout, hidden_act)

    def forward(self, dec_input, enc_words_output, enc_sents_output, self_attn_bias,
                dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                cache=None):
        q = self.pre_process_layer1(dec_input)
        self_attn_output = self.self_attn(q, q, q, self_attn_bias, cache=cache, type='self')
        self_attn_output = self.post_process_layer1(dec_input, self_attn_output)

        q = self.pre_process_layer2(self_attn_output)
        hier_attn_output = self.multi_head_hierarchical_attn(
            q, enc_sents_output, enc_sents_output, enc_words_output, enc_words_output,
            dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
            cache=cache
        )
        hier_attn_output = self.post_process_layer2(self_attn_output, hier_attn_output)

        x = self.pre_process_layer3(hier_attn_output)
        ffd_output = self.pos_wise_ffd(x)
        dec_output = self.post_process_layer3(hier_attn_output, ffd_output)

        return dec_output


class GraphDecoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd, post_process_cmd, device):
        super(GraphDecoder, self).__init__()
        self.n_layers = n_layers

        self.graph_decoder_layers = nn.ModuleList([
            GraphDecoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win,
                pre_post_process_dropout, attn_dropout, relu_dropout,
                hidden_act, pre_process_cmd, post_process_cmd, device
            ) for _ in range(n_layers)
        ])
        self.pre_process_layer = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)

    def forward(self, dec_input, enc_words_output, enc_sents_output,
                dec_self_attn_bias, dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                state=None):
        for i in range(self.n_layers):
            dec_output = self.graph_decoder_layers[i](
                dec_input, enc_words_output, enc_sents_output,
                dec_self_attn_bias, dec_enc_words_attn_bias, dec_enc_sents_attn_bias, graph_attn_bias,
                cache=state.cache['layer_{}'.format(i)] if state is not None and state.cache is not None else None
            )
            dec_input = dec_output

        dec_output = self.pre_process_layer(dec_output)

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
        for l in range(num_layers):
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
            self.cache['layer_{}'.format(l)] = layer_cache

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
