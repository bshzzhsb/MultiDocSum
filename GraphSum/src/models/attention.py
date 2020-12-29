import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, queries, keys, values):
        super(MultiHeadAttention, self).__init__()
        self.keys = queries if keys is None else keys
        self.values = keys if values is None else values
