import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        weight = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) *
                           -(math.log(10000.0) / dim))
        # [1, max_len, dim]
        self.register_buffer('weight', weight)
        self.dim = dim

    def forward(self, pos):
        device = pos.device
        pos_emb = torch.full((*pos.size(), self.dim), 0.0, dtype=torch.float32, device=device)
        if len(pos.size()) == 2:
            pos_emb[:, :, 0::2] = torch.sin(pos.unsqueeze(-1) * self.weight)
            pos_emb[:, :, 1::2] = torch.cos(pos.unsqueeze(-1) * self.weight)
        elif len(pos.size()) == 3:
            pos_emb[:, :, :, 0::2] = torch.sin(pos.unsqueeze(-1) * self.weight)
            pos_emb[:, :, :, 1::2] = torch.cos(pos.unsqueeze(-1) * self.weight)
        return pos_emb


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_inner_hidden, d_hidden, dropout, hidden_act):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_act = hidden_act

        self.fc1 = nn.Linear(d_model, d_inner_hidden)
        self.fc2 = nn.Linear(d_inner_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.hidden_act == "relu":
            hidden = F.relu(hidden)
        elif self.hidden_act == 'tanh':
            hidden = torch.tanh(hidden)

        hidden = self.dropout(hidden)
        out = self.fc2(hidden)

        return out


class PreProcessLayer(nn.Module):

    def __init__(self, d_model, process_cmd, dropout):
        super(PreProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, out):
        for cmd in self.process_cmd:
            if cmd == 'n':
                out = self.layer_norm(out)
            elif cmd == 'd':
                out = self.dropout(out)

        return out


class PostProcessLayer(nn.Module):

    def __init__(self, d_model, process_cmd, dropout):
        super(PostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, prev_out, out):
        for cmd in self.process_cmd:
            if cmd == 'a':
                out = out + prev_out if prev_out is not None else out
            elif cmd == 'n':
                out = self.layer_norm(out)
            elif cmd == 'd':
                out = self.dropout(out)

        return out
