import torch.nn as nn
from SourceCode.ModelModule.SparseSoftmax import Sparsemax


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
            # nn.LayerNorm(self.out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class EmbeddingNetDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
            # nn.LayerNorm(self.out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

