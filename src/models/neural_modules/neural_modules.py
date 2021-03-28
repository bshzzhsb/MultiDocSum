import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_pos):
        super(PositionalEncoding, self).__init__()
        pe = torch.FloatTensor(500, d_pos)
        position = torch.arange(0, 500).unsqueeze(-1)
        weight = torch.exp(torch.arange(0, d_pos, 2, dtype=torch.float) * -(math.log(10000.0) / d_pos))
        pe[:, 0::2] = torch.sin(position * weight)
        pe[:, 1::2] = torch.cos(position * weight)

        self.register_buffer('pe', pe)

    def forward(self, pos):
        pos_embed = F.embedding(pos, self.pe)
        return pos_embed


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        inter = self.dropout_1(F.relu(self.fc1(self.layer_norm(x))))
        output = self.dropout_2(self.fc2(inter))

        output += residual

        return output
