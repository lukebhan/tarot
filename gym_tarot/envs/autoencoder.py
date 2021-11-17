import numpy as np
import torch

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(24, 20), 
                torch.nn.ReLU(),
                torch.nn.Linear(20, 16), 
                torch.nn.ReLU(),
                torch.nn.Linear(16, 12),
                torch.nn.ReLU(),
                torch.nn.Linear(12, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 4))

        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 12),
                torch.nn.ReLU(),
                torch.nn.Linear(12, 16), 
                torch.nn.ReLU(),
                torch.nn.Linear(16, 20), 
                torch.nn.ReLU(),
                torch.nn.Linear(20, 24))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

