from abc import abstractmethod, ABC


class AbstractStaticFactory(ABC):

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def init_MGS(self, prod_env):
        pass


