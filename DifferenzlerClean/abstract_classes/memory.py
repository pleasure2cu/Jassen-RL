from typing import List, Union
import abc


class Memory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def draw_batch(self, size: int):
        pass

    @abc.abstractmethod
    def add_samples(self, xs: List, y: Union[float, int]):
        pass

    @abc.abstractmethod
    def assert_items(self) -> bool:
        pass
