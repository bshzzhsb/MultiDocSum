import torch.nn as nn

from models.neural_modules.attention import \
    MultiHeadAttention, MultiHeadPooling, MultiHeadStructureAttention, GraphAttention
from models.neural_modules.neural_modules import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_k, d_v, d_inner_hidden, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pos_ffd = PositionwiseFeedForward(d_model, d_inner_hidden, dropout)

    def forward(self, k, bias):
        """
        :param k: [batch_size, n_blocks, d_model]
        :param bias:
        :return: [batch_size, n_blocks, d_model]
        """
        attn_input = self.layer_norm(k)

        # [batch_size, n_blocks, d_model]
        attn_output = self.self_attn(attn_input, attn_input, attn_input, bias)
        attn_output = self.dropout(attn_output) + k

        # [batch_size, n_blocks, d_model]
        out = self.pos_ffd(attn_output)

        # [batch_size, n_blocks, d_model]
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner_hidden, dropout):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers

        self.transformer_encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_k, d_v, d_inner_hidden, dropout)
             for _ in range(self.n_layers)]
        )

    def forward(self, enc_input, bias):
        """
        :param enc_input: [batch_size, n_blocks, d_model]
        :param bias: [batch_size * n_blocks, n_heads, n_tokens, n_tokens]
        :return: [batch_size, n_blocks, d_model]
        """
        for i in range(self.n_layers):
            # [batch_size, n_blocks, d_model]
            enc_output = self.transformer_encoder_layers[i](enc_input, bias)

        # [batch_size, n_blocks, d_model]
        return enc_output


class SelfAttentionPoolingLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_v, n_blocks, dropout):
        super(SelfAttentionPoolingLayer, self).__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model

        self.multi_head_pooling = MultiHeadPooling(n_heads, d_model, d_v, dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, bias):
        """
        :param enc_input: [batch_size * n_blocks, n_tokens, d_model]
        :param bias: [batch_size * n_blocks, n_heads, n_tokens, n_tokens]
        :return: [batch_size, n_blocks, d_model]
        """
        key = self.layer_norm(enc_input)

        # [batch_size * n_blocks, d_model]
        attn_output = self.multi_head_pooling(key, key, bias)
        # [batch_size, n_blocks, d_model]
        attn_output = attn_output.contiguous().view(-1, self.n_blocks, self.d_model)

        pooling_output = self.dropout(attn_output)

        # [batch_size, n_blocks, d_model]
        return pooling_output


class GraphEncoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win, dropout):
        super(GraphEncoderLayer, self).__init__()
        self.multi_head_structure_attn = MultiHeadStructureAttention(
            n_heads, d_model, d_k, d_v, pos_win, dropout
        )

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.pos_ffd = PositionwiseFeedForward(d_model, d_inner_hidden, dropout)

    def forward(self, enc_input, bias, graph_attn_bias):
        """
        :param enc_input: [batch_size, n_blocks, d_model]
        :param bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param graph_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        :return: [batch_size, n_blocks, d_model]
        """
        q = self.layer_norm_1(enc_input)
        # [batch_size, n_blocks, d_model]
        attn_output = self.multi_head_structure_attn(q, q, q, bias, graph_attn_bias)
        attn_output = self.dropout_1(attn_output) + enc_input

        # [batch_size, n_blocks, d_model]
        out = self.pos_ffd(attn_output)

        # [batch_size, n_blocks, d_model]
        return out


class GraphEncoder(nn.Module):

    def __init__(self, n_graph_layers, n_heads, n_blocks,
                 d_model, d_k, d_v, d_inner_hidden, pos_win, dropout):
        super(GraphEncoder, self).__init__()
        self.n_graph_layers = n_graph_layers

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.self_attn_pooling_layer = SelfAttentionPoolingLayer(
            n_heads, d_model, d_v, n_blocks, dropout
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
        :param src_words_self_attn_bias: [batch_size * n_blocks, n_heads, n_tokens, d_model]
        :param src_sents_self_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param graph_attn_bias:
        :return: [batch_size, n_blocks, d_model]
        """
        # [batch_size, n_blocks, d_model]
        enc_input = self.self_attn_pooling_layer(enc_words_input, src_words_self_attn_bias)

        for i in range(self.n_graph_layers):
            # [batch_size, n_blocks, d_model]
            enc_output = self.graph_encoder_layers[i](enc_input, src_sents_self_attn_bias, graph_attn_bias)
            enc_input = enc_output

        enc_output = self.layer_norm(enc_output)

        # [batch_size, n_blocks, d_model]
        return enc_output


class GATLayer(nn.Module):

    def __init__(self, n_heads, d_model, dropout, alpha):
        super(GATLayer, self).__init__()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        self.attn = GraphAttention(n_heads, d_model, d_model, dropout, alpha)
        self.out_attn = GraphAttention(1, d_model, d_model, dropout, alpha)

    def forward(self, x, bias):
        attn_in = self.layer_norm_1(x)
        attn = self.attn(attn_in, bias)

        attn = self.dropout_1(attn) + x

        attn_in = self.layer_norm_2(attn)
        attn_out = self.out_attn(attn_in, bias)
        attn_out = self.dropout_2(attn_out) + attn

        return attn_out
