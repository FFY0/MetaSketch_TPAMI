from abc import abstractmethod, ABC
from torch import nn


class AbstractModel(ABC, nn.Module):
    @abstractmethod
    def write(self, input_x, input_y):
        pass

    @abstractmethod
    def query(self, input_x,stream_length):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def normalize_attention_matrix(self):
        pass
