import torch
from torch.utils.data.dataset import T_co

from AbstractClass.TaskRelatedClasses import AbstractMetaTask, AbstractQuerySet, AbstractSupportSet


class MetaTask(AbstractMetaTask):

    def __init__(self, support_set: AbstractSupportSet, query_set: AbstractQuerySet):
        super(MetaTask, self).__init__(support_set, query_set)
        pass

    def to_device(self):
        self.support_set.to_device()
        self.query_set.to_device()

    def to_np(self):
        self.support_set.support_x = self.support_set.support_x.numpy()
        self.support_set.support_y = self.support_set.support_y.numpy()
        self.query_set.query_x = self.query_set.query_x.numpy()
        self.query_set.query_y = self.query_set.query_y.numpy()

    def to_tensor(self):
        self.support_set.support_x = torch.tensor(self.support_set.support_x)
        self.support_set.support_y = torch.tensor(self.support_set.support_y)
        self.query_set.query_x = torch.tensor(self.query_set.query_x)
        self.query_set.query_y = torch.tensor(self.query_set.query_y)

class SupportSet(AbstractSupportSet):

    def __getitem__(self, index) -> T_co:
        return self.support_x[index], self.support_y[index]

    def __init__(self, support_x, support_y, device):
        super().__init__()
        self.support_y = support_y
        self.support_x = support_x
        self.device = device

    def to_device(self):
        self.support_x = self.support_x.to(self.device)
        self.support_y = self.support_y.to(self.device)

    def __len__(self):
        return self.support_y.shape[0]


class QuerySet(AbstractQuerySet):

    def __getitem__(self, index) -> T_co:
        return self.query_x[index], self.query_y[index]

    def __init__(self, query_x, query_y, device):
        super(QuerySet, self).__init__()
        self.query_x = query_x
        self.query_y = query_y
        self.device = device

    def to_device(self):
        self.query_x = self.query_x.to(self.device)
        self.query_y = self.query_y.to(self.device)

    def __len__(self):
        return self.query_y.shape[0]
