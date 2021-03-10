import torch.nn as nn

from modules.attention import MultiHeadAttention, MultiHeadPooling, MultiHeadStructureAttention
from modules.neural_modules import PositionWiseFeedForward, PreProcessLayer, PostProcessLayer


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_k, d_v, d_inner_hidden,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd='n', post_process_cmd='da'):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_process_cmd = pre_process_cmd

        self.pre_process_layer1 = PreProcessLayer(
            d_model, pre_process_cmd, pre_post_process_dropout
        )
        self.pre_process_layer2 = PreProcessLayer(
            d_model, pre_process_cmd, pre_post_process_dropout
        )
        self.pre_process_layer3 = PreProcessLayer(
            d_model, pre_process_cmd, pre_post_process_dropout
        )
        self.post_process_layer1 = PostProcessLayer(
            d_model, post_process_cmd, pre_post_process_dropout
        )
        self.post_process_layer2 = PostProcessLayer(
            d_model, post_process_cmd, pre_post_process_dropout
        )
        self.pos_wise_ffd = PositionWiseFeedForward(
            d_model, d_inner_hidden, d_model, relu_dropout, hidden_act
        )
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, attn_dropout)

    def forward(self, q, k, bias):
        """
        :param q: [batch_size, n_blocks, d_model]
        :param k: [batch_size, n_blocks, d_model]
        :return: [batch_size, n_blocks, d_model]
        """
        k = self.pre_process_layer1(k) if k else None
        v = k if k else None

        # [batch_size, n_blocks, d_model]
        attn_output = self.self_attn(self.pre_process_layer2(q), k, v, bias)
        attn_output = self.post_process_layer1(q, attn_output)

        # [batch_size, n_blocks, d_model]
        ffd_output = self.pos_wise_ffd(self.pre_process_layer3(attn_output))
        out = self.post_process_layer2(attn_output, ffd_output)

        # [batch_size, n_blocks, d_model]
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers, with_post_process,
                 n_heads, d_k, d_v, d_model, d_inner_hidden,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd='n', post_process_cmd='da'):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.with_post_process = with_post_process

        self.transformer_encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_k, d_v, d_inner_hidden,
                                     pre_post_process_dropout, attn_dropout, relu_dropout,
                                     hidden_act, pre_process_cmd, post_process_cmd)
             for i in range(self.n_layers)]
        )
        self.pre_process_layer = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)

    def forward(self, enc_input, bias):
        """
        :param enc_input: [batch_size, n_blocks, d_model]
        :return: [batch_size, n_blocks, d_model]
        """
        for i in range(self.n_layers):
            # [batch_size, n_blocks, d_model]
            enc_output = self.transformer_encoder_layers[i](enc_input, None, bias)

        if self.with_post_process:
            enc_output = self.pre_process_layer(enc_output)

        # [batch_size, n_blocks, d_model]
        return enc_output


class SelfAttentionPoolingLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_v, n_blocks,
                 pre_post_process_dropout, attn_dropout,
                 pre_process_cmd='n'):
        super(SelfAttentionPoolingLayer, self).__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model

        self.pre_process_layer = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.multi_head_pooling = MultiHeadPooling(n_heads, d_model, d_v, attn_dropout)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, enc_input, bias):
        """
        :param enc_input: [batch_size * n_blocks, n_tokens, d_model]
        :param bias: [batch_size * n_blocks, n_heads, n_tokens, n_tokens]
        :return: [batch_size, n_blocks, d_model]
        """
        key = self.pre_process_layer(enc_input)

        # [batch_size * n_blocks, d_model]
        attn_output = self.multi_head_pooling(key, key, bias)
        # [batch_size, n_blocks, d_model]
        attn_output = attn_output.contiguous().view(-1, self.n_blocks, self.d_model)

        pooling_output = self.dropout(attn_output)

        # [batch_size, n_blocks, d_model]
        return pooling_output


class GraphEncoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd='n', post_process_cmd='da'):
        super(GraphEncoderLayer, self).__init__()
        self.multi_head_structure_attn = MultiHeadStructureAttention(
            n_heads, d_model, d_k, d_v, pos_win, attn_dropout
        )
        self.pre_process_layer1 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.pre_process_layer2 = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)
        self.post_process_layer1 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)
        self.post_process_layer2 = PostProcessLayer(d_model, post_process_cmd, pre_post_process_dropout)
        self.pos_wise_ffd = PositionWiseFeedForward(d_model, d_inner_hidden, d_model, relu_dropout, hidden_act)

    def forward(self, enc_input, bias, graph_attn_bias):
        """
        :param enc_input: [batch_size, n_blocks, d_model]
        :return: [batch_size, n_blocks, d_model]
        :param bias: [batch_size, n_heads, n_blocks, n_blocks]
        :param graph_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        """
        q = self.pre_process_layer1(enc_input)
        # [batch_size, n_blocks, d_model]
        attn_output = self.multi_head_structure_attn(q, q, q, bias, graph_attn_bias)
        attn_output = self.post_process_layer1(enc_input, attn_output)

        # [batch_size, n_blocks, d_model]
        ffd_output = self.pos_wise_ffd(self.pre_process_layer2(attn_output))

        out = self.post_process_layer2(attn_output, ffd_output)

        # [batch_size, n_blocks, d_model]
        return out


class GraphEncoder(nn.Module):

    def __init__(self, n_graph_layers, n_heads, n_blocks,
                 d_model, d_k, d_v, d_inner_hidden, pos_win,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd='n', post_process_cmd='da'):
        super(GraphEncoder, self).__init__()
        self.n_graph_layers = n_graph_layers

        self.self_attn_pooling_layer = SelfAttentionPoolingLayer(
            n_heads, d_model, d_v, n_blocks, pre_post_process_dropout, attn_dropout
        )
        self.graph_encoder_layers = nn.ModuleList(
            [GraphEncoderLayer(
                n_heads, d_model, d_k, d_v, d_inner_hidden, pos_win,
                pre_post_process_dropout, attn_dropout, relu_dropout,
                hidden_act, pre_process_cmd, post_process_cmd
            ) for i in range(n_graph_layers)]
        )
        self.pre_process_layer = PreProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)

    def forward(self, enc_words_input, src_words_self_attn_bias,
                src_sents_self_attn_bias, graph_attn_bias):
        """
        :param enc_words_input: [batch_size * n_blocks, n_tokens, d_model]
        :param src_words_self_attn_bias: [batch_size * n_blocks, n_heads, n_tokens, d_model]
        :param src_sents_self_attn_bias: [batch_size, n_heads, n_blocks, n_blocks]
        :return: [batch_size, n_blocks, d_model]
        """
        # [batch_size, n_blocks, d_model]
        sents_vec = self.self_attn_pooling_layer(enc_words_input, src_words_self_attn_bias)
        enc_input = sents_vec

        for i in range(self.n_graph_layers):
            # [batch_size, n_blocks, d_model]
            enc_output = self.graph_encoder_layers[i](enc_input, src_sents_self_attn_bias, graph_attn_bias)
            enc_input = enc_output

        enc_output = self.pre_process_layer(enc_output)

        # [batch_size, n_blocks, d_model]
        return enc_output
