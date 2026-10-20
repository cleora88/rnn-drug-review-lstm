"""Manual cells following the assignment equations; no built-in LSTM."""
import torch
from torch import nn

class Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kind):
        super().__init__()
        self.kind, self.hidden_size = kind, hidden_size
        gates = 3 if kind == 'cifg' else 4
        self.x = nn.Linear(input_size, gates * hidden_size)
        self.h = nn.Linear(hidden_size, gates * hidden_size, bias=False)
        if kind == 'wmc':
            self.ci = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cf = nn.Linear(hidden_size, hidden_size, bias=False)
            self.co = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        gates = (self.x(x) + self.h(h)).chunk(3 if self.kind == 'cifg' else 4, -1)
        if self.kind == 'cifg':
            f, o, g = gates
            f = f.sigmoid()
            i = 1 - f
        else:
            i, f, o, g = gates
            if self.kind == 'wmc':
                i = i + self.ci(c).tanh()
                f = f + self.cf(c).tanh()
            i, f = i.sigmoid(), f.sigmoid()
        c = f * c + i * g.tanh()
        if self.kind == 'wmc':
            o = o + self.co(c).tanh()
        return o.sigmoid() * c.tanh(), c

class Classifier(nn.Module):
    def __init__(self, vocab_size, kind, embedding=64, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.cell = Cell(embedding, hidden, kind)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden, 3)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        h = embedded.new_zeros(tokens.size(0), self.cell.hidden_size)
        c = torch.zeros_like(h)
        for t in range(tokens.size(1)):
            hn, cn = self.cell(embedded[:, t], (h, c))
            mask = (tokens[:, t] != 0).unsqueeze(1)
            h, c = torch.where(mask, hn, h), torch.where(mask, cn, c)
        return self.output(self.dropout(h))
