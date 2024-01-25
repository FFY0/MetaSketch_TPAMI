from abc import abstractmethod, ABC


class AbstractLogger(ABC):
    @abstractmethod
    def logging(self,model,step):
        pass
