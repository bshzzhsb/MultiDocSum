import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_inner_hidden, d_hidden, dropout, hidden_act):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_inner_hidden)
        self.fc2 = nn.Linear(d_inner_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.dropout(self.fc1(x))
        out = self.fc2(hidden)

        return out


class PrePostProcessLayer(nn.Module):

    def __init__(self, d_model, process_cmd, dropout):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, prev_out, out):
        for cmd in self.process_cmd:
            if cmd == 'a':
                out = out + prev_out if prev_out else out
            elif cmd == 'n':
                out = self.layer_norm(out)
            elif cmd == 'd':
                out = self.dropout(out)

        return out
