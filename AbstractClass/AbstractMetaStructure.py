from abc import abstractmethod, ABC
from torch import nn


class AbstractMetaStructure(ABC):
    pass
    # @abstractmethod
    # def train(self, train_config):
    #     pass


class AbstractLossFunc(ABC, nn.Module):
    pass
