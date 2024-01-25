import torch
import torch.nn as nn


class RefineNet(nn.Module):
    def __init__(self, source_embedding_dim, source_refined_dim, source_refined_hidden_layer_size):
        super().__init__()
        self.in_dim = source_embedding_dim
        self.out_dim = source_refined_dim
        self.hidden_layer_size = source_refined_hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
        )

    def forward(self, x):
        return self.net(x)

