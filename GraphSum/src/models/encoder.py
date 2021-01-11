import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from neural_modules import PositionWiseFeedForward, PrePostProcessLayer


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_k, d_v, d_inner_hidden, bias,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd='n', post_process_cmd='da'):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_process_cmd = pre_process_cmd

        self.pre_process_layer1 = PrePostProcessLayer(
            d_model, pre_process_cmd, pre_post_process_dropout
        )
        self.pre_process_layer2 = PrePostProcessLayer(
            d_model, pre_process_cmd, pre_post_process_dropout
        )
        self.pre_process_layer3 = PrePostProcessLayer(
            d_model, pre_process_cmd, pre_post_process_dropout
        )
        self.post_process_layer1 = PrePostProcessLayer(
            d_model, post_process_cmd, pre_post_process_dropout
        )
        self.post_process_layer2 = PrePostProcessLayer(
            d_model, post_process_cmd, pre_post_process_dropout
        )
        self.pos_wise_ffd = PositionWiseFeedForward(
            d_model, d_inner_hidden, d_model, relu_dropout, hidden_act
        )
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, bias, attn_dropout)

    def forward(self, q, k):
        k = self.pre_process_layer1(None, k) if k else None
        v = k if k else None
        attn_output = self.self_attn(self.pre_process_layer2(None, q), k, v)
        attn_output = self.post_process_layer1(q, attn_output)
        ffd_output = self.pos_wise_ffd(self.pre_process_layer3(None, attn_output))
        out = self.post_process_layer2(attn_output, ffd_output)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers, with_post_process,
                 n_heads, d_k, d_v, d_model, d_inner_hidden, bias,
                 pre_post_process_dropout, attn_dropout, relu_dropout,
                 hidden_act, pre_process_cmd='n', post_process_cmd='da'):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.with_post_process = with_post_process

        self.transformer_encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_k, d_v, d_inner_hidden, bias,
                                     pre_post_process_dropout, attn_dropout, relu_dropout,
                                     hidden_act, pre_process_cmd, post_process_cmd)
             for i in range(self.n_layers)]
        )
        self.pre_process_layer = PrePostProcessLayer(d_model, pre_process_cmd, pre_post_process_dropout)

    def forward(self, enc_input):
        enc_output = None
        for i in range(self.n_layers):
            enc_output = self.transformer_encoder_layers[i](enc_input, None)

        if self.with_post_process:
            enc_output = self.pre_process_layer(None, enc_output)

        return enc_output
