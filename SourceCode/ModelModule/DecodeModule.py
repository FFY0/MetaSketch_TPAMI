import torch
import torch.nn as nn


class WeightDecodeNet(nn.Module):
    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size
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


class WeightDecodeNetResidual(nn.Module):
    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size
        self.hidden_1 = nn.Linear(self.in_dim, self.hidden_layer_size)
        self.activate_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(self.hidden_layer_size, self.in_dim)
        self.activate_2 = nn.ReLU()
        self.hidden_3 = nn.Linear(self.in_dim, (self.in_dim + self.out_dim) // 2)
        self.activate_3 = nn.ReLU()
        self.out = nn.Linear((self.in_dim + self.out_dim) // 2, self.out_dim)

    def forward(self, x):
        y = (self.hidden_1(x))
        y = self.activate_1(y)
        y = (self.activate_2((self.hidden_2(y)) + x))
        y = (self.activate_3((self.hidden_3(y))))
        return self.out(y)


class ResExistDecodeNet(nn.Module):
    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size
        self.hidden_1 = nn.Linear(self.in_dim, self.hidden_layer_size)
        self.ln1 = nn.LayerNorm(self.hidden_layer_size)

        self.activate_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(self.hidden_layer_size, self.in_dim)
        self.activate_2 = nn.ReLU()
        self.hidden_3 = nn.Linear(self.in_dim, (self.in_dim + self.out_dim) // 2)
        self.activate_3 = nn.ReLU()
        self.ln3 = nn.LayerNorm((self.in_dim + self.out_dim) // 2)
        self.out = nn.Linear((self.in_dim + self.out_dim) // 2, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.hidden_1(x)
        y = self.activate_1(self.ln1(y))
        y = self.activate_2(self.hidden_2(y) + x)
        y = self.activate_3(self.ln3(self.hidden_3(y)))
        y = self.out(y)

        return self.sigmoid(y)
