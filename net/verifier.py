import os
import torch
import torch.nn as nn
import types
import torch.nn.functional as F

__all__ = ["Verifier"]


class Verifier(nn.Module):
    def __init__(self, N, num_classes):
        super(Verifier, self).__init__()
        self.input_size = int(num_classes * N)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    net = Verifier(N=50, num_classes=1000)
    prob = torch.randn(10, 1000)
    pred = net(prob)
    print(pred)