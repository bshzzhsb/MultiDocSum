import torch.nn as nn

from model_gat.neural_modules.attention import GraphAttention
from models.neural_modules.neural_modules import PositionwiseFeedForward
from models.layers.encoder import SelfAttentionPoolingLayer, GraphEncoderLayer


class GraphEncoder(nn.Module):

    def __init__(self, n_graph_layers, n_heads, n_blocks,
                 d_model, d_k, d_v, d_inner_hidden, pos_win, dropout):
        super(GraphEncoder, self).__init__()
        self.n_graph_layers = n_graph_layers

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.self_attn_pooling_layer = SelfAttentionPoolingLayer(
            n_heads, d_model, d_v, n_blocks, dropout
        )
        self.graph_attn_layers = nn.ModuleList(
            [GATLayer(n_heads, d_model, d_inner_hidden, dropout, 0.2) for _ in range(2)]
        )
        self.graph_encoder_layers = nn.ModuleList(
            [GraphEncoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win, dropout
            ) for _ in range(n_graph_layers)]
        )

    def forward(self, enc_words_input, src_words_self_attn_bias,
                src_sents_self_attn_bias, graph_attn_bias):
        """
        :param enc_words_input: [batch_size * n_blocks, n_tokens, d_model]
        :param src_words_self_attn_bias: [batch_size * n_blocks, n_heads, n_tokens, n_tokens]
        :param src_sents_self_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param graph_attn_bias:
        :return: [batch_size, n_blocks, d_model]
        """
        # [batch_size, n_blocks, d_model]
        enc_input = self.self_attn_pooling_layer(enc_words_input, src_words_self_attn_bias)

        for i in range(2):
            enc_input = self.graph_attn_layers[i](enc_input, src_sents_self_attn_bias)

        for i in range(self.n_graph_layers):
            # [batch_size, n_blocks, d_model]
            enc_output = self.graph_encoder_layers[i](enc_input, src_sents_self_attn_bias, graph_attn_bias)
            enc_input = enc_output

        enc_output = self.layer_norm(enc_output)

        # [batch_size, n_blocks, d_model]
        return enc_output


class GATLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_inner_hidden, dropout, alpha):
        super(GATLayer, self).__init__()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hidden, dropout)

        self.attn = GraphAttention(n_heads, d_model, d_model, dropout, alpha)
        self.out_attn = GraphAttention(1, d_model, d_model, dropout, alpha)

    def forward(self, x, bias):
        attn_in = self.layer_norm_1(x)
        attn = self.attn(attn_in, bias)

        attn = self.dropout_1(attn) + x

        attn_in = self.layer_norm_2(attn)
        attn_out = self.out_attn(attn_in, bias)
        attn_out = self.dropout_2(attn_out) + attn

        out = self.pos_ffn(attn_out)
        return out
