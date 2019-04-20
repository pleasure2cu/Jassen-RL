import random
from typing import List, Tuple

from helper_functions import Sample


# see https://arxiv.org/pdf/1312.5602v1.pdf
class ReplayMemory:

    _size: int
    _items: List[Sample]

    def __init__(self, size: int):
        self._size = size
        self._items = []

    def draw_samples(self, n: int):
        assert n >= 0
        n = min(n, len(self._items))
        return random.sample(self._items, n)

    def add_samples(self, samples: List[Tuple]):
        self._items += samples
        s = min(self._size, len(self._items))
        self._items = self._items[-s:]
