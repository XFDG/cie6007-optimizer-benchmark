"""Small MLP; distinguish input, hidden, and classifier matrices explicitly."""
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.input = nn.Linear(784, hidden)
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.head(self.body(self.input(x.flatten(1)).relu()))
