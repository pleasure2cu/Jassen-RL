import numpy as np
import random
from typing import List


# see https://arxiv.org/pdf/1312.5602v1.pdf
from Sample import RnnSample


class ReplayMemory:

    _size: int
    _items: List
    _index: int

    def __init__(self, size: int):
        self._size = size
        self._items = []
        self._index = 0

    def draw_samples(self, n: int):
        assert n >= 0
        n = min(n, len(self._items))
        return random.sample(self._items, n)

    def add_samples(self, samples: List):
        self._items += samples
        s = min(self._size, len(self._items))
        self._items = self._items[-s:]


class RnnReplayMemory(ReplayMemory):

    _size: int
    _items: List[List[RnnSample]]

    def __init__(self, size: int):
        super().__init__(size)
        self._items = [[] for _ in range(9)]

    def draw_samples(self, n: int) -> List[RnnSample]:
        assert n >= 0
        time_series_class = np.random.randint(9)
        n = min(n, len(self._items[time_series_class]))
        return random.sample(self._items[time_series_class], n)

    def add_samples(self, samples: List[RnnSample]):
        for sample in samples:
            self._items[sample.number_of_time_steps()-1].append(sample)
        for i in range(len(self._items)):
            if len(self._items[i]) > self._size:
                self._items[i] = self._items[i][-self._size:]



