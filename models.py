"""Vanilla, CIFG and WMC recurrent cells for review classification."""

import torch
from torch import Tensor, nn

CELL_TYPES = ("vanilla", "cifg", "wmc")
RecurrentState = tuple[Tensor, Tensor]


class Cell(nn.Module):
    """Compute one recurrent step using the project appendix equations."""

    def __init__(self, input_size: int, hidden_size: int, kind: str) -> None:
        super().__init__()
        if kind not in CELL_TYPES:
            raise ValueError(f"Unknown cell type {kind!r}; choose from {CELL_TYPES}.")
        self.kind = kind
        self.hidden_size = hidden_size
        self.gate_count = 3 if kind == "cifg" else 4
        self.x = nn.Linear(input_size, self.gate_count * hidden_size)
        self.h = nn.Linear(hidden_size, self.gate_count * hidden_size, bias=False)
        if kind == "wmc":
            self.ci = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cf = nn.Linear(hidden_size, hidden_size, bias=False)
            self.co = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, state: RecurrentState) -> RecurrentState:
        hidden, memory = state
        projections = (self.x(x) + self.h(hidden)).chunk(self.gate_count, dim=-1)
        if self.kind == "cifg":
            forget_gate, output_gate, candidate = projections
            forget_gate = forget_gate.sigmoid()
            input_gate = 1 - forget_gate
        else:
            input_gate, forget_gate, output_gate, candidate = projections
            if self.kind == "wmc":
                input_gate = input_gate + self.ci(memory).tanh()
                forget_gate = forget_gate + self.cf(memory).tanh()
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()

        memory = forget_gate * memory + input_gate * candidate.tanh()
        if self.kind == "wmc":
            # The output gate reads the updated memory c_t.
            output_gate = output_gate + self.co(memory).tanh()
        hidden = output_gate.sigmoid() * memory.tanh()
        return hidden, memory


class Classifier(nn.Module):
    """Classify a review using its last non-padding recurrent state."""

    def __init__(
        self, vocab_size: int, kind: str, embedding: int = 64, hidden: int = 64
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.cell = Cell(embedding, hidden, kind)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden, 3)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        hidden = embedded.new_zeros(tokens.size(0), self.cell.hidden_size)
        memory = torch.zeros_like(hidden)
        for position in range(tokens.size(1)):
            next_hidden, next_memory = self.cell(
                embedded[:, position], (hidden, memory)
            )
            valid_token = (tokens[:, position] != 0).unsqueeze(1)
            hidden = torch.where(valid_token, next_hidden, hidden)
            memory = torch.where(valid_token, next_memory, memory)
        return self.output(self.dropout(hidden))
