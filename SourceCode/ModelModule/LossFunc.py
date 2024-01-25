import torch

from AbstractClass.AbstractMetaStructure import AbstractLossFunc
from torch import nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2, weights=None, device=None):
        self.weights = weights
        learnable = True
        if weights is not None:
            learnable = False
        self.learnable = learnable
        super(AutomaticWeightedLoss, self).__init__()
        if learnable:
            params = torch.ones(num, requires_grad=learnable)
            self.params = torch.nn.Parameter(params)
        else:
            self.params = torch.tensor(weights, requires_grad=learnable, device=device)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class LossFunc_for_ARE_AAE(AbstractLossFunc):
    def __init__(self):
        super().__init__()
        self.auto_weighted_loss = AutomaticWeightedLoss(2)

    def forward(self, pred, y):
        are = torch.mean(torch.abs((pred - y) / y))
        aae = torch.mean(torch.abs((pred - y)))
        return self.auto_weighted_loss(aae, are)


class LossFunc_for_MSE_ARE(AbstractLossFunc):
    def __init__(self):
        super().__init__()
        self.auto_weighted_loss = AutomaticWeightedLoss(2)
        self.mse_func = torch.nn.MSELoss()

    def forward(self, pred, y):
        are = torch.mean(torch.abs((pred - y) / y))
        mse = self.mse_func(pred, y)
        loss = self.auto_weighted_loss(mse, are)
        return loss
class LossFunc_exist_for_BCE(AbstractLossFunc):
    def __init__(self):
        super().__init__()
        self.bce_func = torch.nn.BCELoss()

    def forward(self, pred, y):
        bce = self.bce_func(pred, y)
        return bce
