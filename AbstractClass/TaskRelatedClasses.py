from abc import abstractmethod, ABC
from torch.utils.data import Dataset, DataLoader


class AbstractSupportSet(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, index):
        pass

    def __init__(self):
        super(AbstractSupportSet, self).__init__()
        self.support_x = None
        self.support_y = None

    @abstractmethod
    def __len__(self):
        pass


class AbstractQuerySet(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, index):
        pass

    def __init__(self):
        super(AbstractQuerySet, self).__init__()
        self.query_x = None
        self.query_y = None
        pass

    @abstractmethod
    def __len__(self):
        pass


class AbstractSupportDataLoader(DataLoader, ABC):
    @abstractmethod
    def __init__(self, support_set: AbstractSupportSet, dataset):
        super().__init__(dataset)
        self.support_set = support_set


class AbstractQueryDataLoader(DataLoader, ABC):
    def __init__(self, query_set: AbstractQuerySet, dataset):
        super().__init__(dataset)
        self.query_set = query_set


class AbstractMetaTask(ABC):

    @abstractmethod
    def __init__(self, support_set: AbstractSupportSet, query_set: AbstractQuerySet):
        self.support_set = support_set
        self.query_set = query_set


class AbstractTaskConsumer(ABC):
    @abstractmethod
    def consume_train_task(self, q, dev) -> AbstractMetaTask:
        pass

    def del_meta_task(self, meta_task):
        pass


# support set 生成
class SupportGeneratorInterface(ABC):

    def __init__(self):
        self.device = None
        pass

    def set_device(self, device):
        self.device = device

    @abstractmethod
    def flush_tensor(self):
        pass
    @abstractmethod
    def sample_train_support(self, item_size=None, skew_ratio=None):
        pass

    @abstractmethod
    def sample_test_support(self, item_size=None, skew_ratio=None):
        pass


#
# class AbstractBaseSupportGenerator(SupportGeneratorInterface, ABC):
#     @abstractmethod
#     def sample_train_support(self):
#         pass
#
#     @abstractmethod
#     def sample_test_support(self):
#         pass


# 装饰器
class AbstractDecoratorSupportGenerator(SupportGeneratorInterface, ABC):
    def __init__(self, base_support_generator: SupportGeneratorInterface):
        super().__init__()
        self.base_support_generator = base_support_generator

    def flush_tensor(self):
        self.base_support_generator.flush_tensor()

    def set_device(self, device):
        self.device = device
        self.base_support_generator.set_device(device)
    # @abstractmethod
    # def decorate_train_support(self, item, item_frequency):
    #     pass
    #
    # @abstractmethod
    # def decorate_test_support(self):
    #     pass


# query set 生成
class AbstractQueryGenerator(ABC):
    def __init__(self):
        self.device = None
        pass

    def set_device(self, device):
        self.device = device

    @abstractmethod
    def generate_train_query(self, support_x, support_y, stream_node_vec_list):
        pass

    @abstractmethod
    def generate_test_query(self, support_x, support_y, stream_node_vec_list):
        pass


# meta task 生成
class AbstractTaskProducer(ABC):
    def __init__(self, support_generator: SupportGeneratorInterface, query_generator: AbstractQueryGenerator):
        self.support_generator = support_generator
        self.query_generator = query_generator

    @abstractmethod
    def produce_train_task(self, q, dev):
        pass

    @abstractmethod
    def produce_test_task(self):
        pass
