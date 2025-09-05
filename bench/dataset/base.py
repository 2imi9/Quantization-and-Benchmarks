import abc
from datasets import Dataset


class BaseDataset(abc.ABC):

    DATASET_NAME = ""

    @classmethod
    @abc.abstractmethod
    def load(cls, debug: bool = False) -> Dataset:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def measure(cls, predictions: list) -> dict:
        pass
