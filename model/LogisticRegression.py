import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        output = self.linear(x)
        return output, x


class Cifar10LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32, 10)

    def forward(self, x):
        output = self.linear(x)
        return output, x