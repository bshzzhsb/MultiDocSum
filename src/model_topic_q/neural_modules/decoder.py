import torch.nn as nn

from model_topic_q.neural_modules.attention import MultiHeadTopicAttention
from models.layers.encoder import SelfAttentionPoolingLayer
from models.layers.decoder import GraphDecoderLayer, GraphDecoderState


class GraphDecoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner_hidden,
                 tgt_len, pos_win, dropout, device, topic=None):
        super(GraphDecoder, self).__init__()
        self.n_layers = n_layers
        self.topic = topic

        if self.topic == 'sum':
            self.pooling = SelfAttentionPoolingLayer(n_heads, d_model, d_v, tgt_len, dropout)
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
            dec_input = self.pooling(dec_input, None)

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
