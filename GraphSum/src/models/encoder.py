import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_k, d_v):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)

