import random
from functools import reduce
from typing import List, Union, Tuple

import numpy as np

from abstract_classes.memory import Memory


class ReplayMemory(Memory):

    _size: int
    _items: List[Tuple[np.ndarray, Union[float, int]]]

    def __init__(self, size: int):
        self._size = size
        self._items = []

    def draw_batch(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        n = min(size, len(self._items))
        sample_tuples = random.sample(self._items, n)
        x_batch, y_batch = zip(*sample_tuples)
        return np.array(x_batch), np.array(y_batch)  # TODO: test

    def add_samples(self, xs: List, y: Union[float, int]):
        self._items += [(np.reshape(x, -1), y) for x in xs]
        s = min(self._size, len(self._items))
        self._items = self._items[-s:]

    def assert_items(self) -> bool:
        this_round = self._items[-4:]
        assert np.all(reduce(lambda x, y: x + y[0][:36], this_round, np.zeros(36)) == 1)
        for i, sample in enumerate(this_round):
            assert sample[0][-1] == i
        return True


class RnnReplayMemory(Memory):

    _class_size: int
    _items: List[List[Tuple[np.ndarray, np.ndarray, Union[float, int]]]]

    def __init__(self, size: int):
        self._class_size = int(size / 9 + 0.5)
        self._items = [[] for _ in range(9)]

    def draw_batch(self, size: int) -> Tuple[List[np.ndarray], np.ndarray]:
        series_length_index = np.random.randint(9)
        n = min(size, len(self._items[series_length_index]))
        sample_tuples = random.sample(self._items[series_length_index], n)
        rnn_batch, aux_batch, y_batch = zip(*sample_tuples)
        return [np.array(rnn_batch), np.array(aux_batch)], np.array(y_batch)

    def add_samples(self, xs: List[Tuple[np.ndarray, np.ndarray]], y: Union[float, int]):
        for x in xs:
            self._items[len(x[0]) - 1].append((*x, y))  # index is decided by the nbr of time steps in the RNN input
        self._items = [
            ts_list if len(ts_list) <= self._class_size else ts_list[-self._class_size:] for ts_list in self._items
        ]

    def assert_items(self) -> bool:
        for player_i in range(1, 5):
            for series_i in range(1, 9):
                tuple_before = self._items[series_i - 1][-player_i]
                tuple_now = self._items[series_i][-player_i]
                # check that the y is always the same
                assert tuple_before[-1] == tuple_now[-1]
                # check that the remaining points are decreasing
                assert tuple_before[1][-3] >= tuple_now[1][-3]
                # check that the whole history is the same as before plus something
                old = tuple_before[0][:series_i - 1]
                new = tuple_now[0][:series_i - 1]
                assert np.array_equal(old, new)
        return True
