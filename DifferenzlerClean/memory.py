import random
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
        return np.array(x_batch), np.array(y_batch)

    def add_samples(self, xs: List, y: Union[float, int]):
        self._items += [(np.reshape(x, -1), y) for x in xs]
        s = min(self._size, len(self._items))
        self._items = self._items[-s:]

    def save_memory(self, name_base: str, folder_path: str = './'):
        if not folder_path.endswith('/'):
            folder_path += '/'
        x_matrix = np.array([t[0] for t in self._items])
        y_vector = np.array([t[1] for t in self._items])
        np.save(folder_path + 'x_' + name_base, x_matrix)
        np.save(folder_path + 'y_' + name_base, y_vector)

    def load_memory(self, name_base: str, folder_path: str = './'):
        print("loading {} ReplayMemory".format(name_base))
        if not folder_path.endswith('/'):
            folder_path += '/'
        x_matrix = np.load(folder_path + 'x_' + name_base + '.npy')
        y_vector = np.load(folder_path + 'y_' + name_base + '.npy')
        assert len(x_matrix) == len(y_vector), (len(x_matrix), len(y_vector))
        self._items = [(x_matrix[i], y_vector[i]) for i in range(len(x_matrix))]
        s = min(self._size, len(self._items))
        self._items = self._items[-s:]

    def assert_items(self) -> bool:
        block = self._items[-360:]
        for turn_i in range(12):
            base_sample = block[turn_i]
            for sample_offset in range(turn_i + 12, 360, 12):
                test_sample = block[sample_offset]
                if np.array_equal(base_sample[0], test_sample[0]):
                    assert False
                if base_sample[1] != base_sample[1]:
                    assert False
        return True


class RnnReplayMemory(Memory):
    _nbr_of_sublists = 8
    _class_size: int
    _items: List[List[Tuple[np.ndarray, np.ndarray, Union[float, int]]]]

    def __init__(self, size: int):
        self._class_size = int(size / RnnReplayMemory._nbr_of_sublists + 0.5)
        self._items = [[] for _ in range(RnnReplayMemory._nbr_of_sublists)]

    def draw_batch(self, size: int) -> Tuple[List[np.ndarray], np.ndarray]:
        series_length_index = np.random.randint(RnnReplayMemory._nbr_of_sublists)
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

    def save_memory(self, name_base: str, folder_path: str = './'):
        if not folder_path.endswith('/'):
            folder_path += '/'
        rnn_tensors = [np.array([t[0] for t in bucket]) for bucket in self._items]
        aux_tensor = np.array([np.array([t[1] for t in bucket]) for bucket in self._items])
        y_tensor = np.array([np.array([t[2] for t in bucket]) for bucket in self._items])
        for i, rnn_tensor in enumerate(rnn_tensors):
            np.save(folder_path + "rnn_{}_".format(i) + name_base, rnn_tensor)
        np.save(folder_path + "aux_" + name_base, aux_tensor)
        np.save(folder_path + "y_" + name_base, y_tensor)

    def load_memory(self, name_base: str, folder_path: str = './'):
        print("loading {} RnnReplayMemory".format(name_base))
        if not folder_path.endswith('/'):
            folder_path += '/'
        rnn_tensors: np.ndarray = [np.load(folder_path + 'rnn_{}_'.format(i) + name_base + '.npy') for i in range(self._nbr_of_sublists)]
        aux_tensor: np.ndarray = np.load(folder_path + 'aux_' + name_base + '.npy')
        y_tensor: np.ndarray = np.load(folder_path + 'y_' + name_base + '.npy')
        for bucket_i in range(self._nbr_of_sublists):
            rnn_part, aux_part, y_part = rnn_tensors[bucket_i], aux_tensor[bucket_i], y_tensor[bucket_i]
            assert len(rnn_part) == len(aux_part) and len(aux_part) == len(y_part)
            for i in range(len(y_part)):
                self._items[bucket_i].append((rnn_part[i], aux_part[i], y_part[i]))
        self._items = [
            ts_list if len(ts_list) <= self._class_size else ts_list[-self._class_size:] for ts_list in self._items
        ]

    def assert_items(self) -> bool:
        for bucket in self._items:
            block = bucket[-360:]

            # group all the samples that should be the same
            grouped = [
                [
                    block[sample_offset] for sample_offset in range(turn_i, 360, 12)
                ]
                for turn_i in range(12)
            ]

            for group in grouped:
                base_sample = group[0]
                for test_sample in group[1:]:
                    if not np.array_equal(base_sample[0], test_sample[0]):
                        assert False
                    if not np.array_equal(base_sample[1], test_sample[1]):
                        assert False
                    if base_sample[2] != test_sample[2]:
                        assert False

        # check the rewards across buckets
        a = [
            np.array([sample[2] for sample in bucket])
            for bucket in self._items
        ]
        for test_sample in a[1:]:
            if not np.array_equal(a[0], test_sample):
                assert False

        return True
