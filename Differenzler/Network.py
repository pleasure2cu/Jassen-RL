from typing import List

from Memory import ReplayMemory, RnnReplayMemory
import keras
import numpy as np

from helper_functions import turn_rnn_samples_into_batch


class Network:
    _replay_memory: ReplayMemory
    _neural_network: keras
    _batch_size: int
    _can_train: bool

    def __init__(self, neural_network: keras, memory: ReplayMemory, batch_size: int, can_train: bool):
        self._neural_network = neural_network
        self._replay_memory = memory
        self._batch_size = batch_size
        self._can_train = can_train

    def evaluate(self, network_input: np.ndarray) -> np.array:
        pass

    def add_samples(self, samples: List):
        self._replay_memory.add_samples(samples)

    def train(self) -> float:
        if not self._can_train:
            return -1
        samples = self._replay_memory.draw_samples(self._batch_size)
        n = len(samples)
        if n == 0:
            return -1
        x = np.zeros((n, samples[0][0].size))
        y = np.zeros(n)
        for i in range(n):
            x[i] = samples[i][0]
            y[i] = samples[i][1]
        return self._neural_network.train_on_batch(x, y)

    def save_network(self, file_path: str):
        self._neural_network.save(file_path)


class PredictionNetwork(Network):
    def evaluate(self, network_input: np.array) -> int:
        output = self._neural_network.predict(np.reshape(network_input, (1, -1)))
        return int(output[0][0] + 0.5)


class RnnStrategyNetwork(Network):
    _replay_memory: RnnReplayMemory

    def train(self) -> float:
        if not self._can_train:
            return -1
        samples = self._replay_memory.draw_samples(self._batch_size)
        n = len(samples)
        if n == 0:
            return -1
        x_rnn, x_aux, y = turn_rnn_samples_into_batch(samples)
        return self._neural_network.train_on_batch([x_rnn, x_aux], y)

    def evaluate(self, network_input: List[np.ndarray]) -> np.ndarray:
        """
        :param network_input: has two entries, the first one needs to be 3 dimensional (input for RNN part). The second
                part needs to be two dimensional (input for the auxiliary part)
        :return: the returned values as vector (1D)
        """
        assert len(network_input[0].shape) == 3
        assert len(network_input[1].shape) == 2
        output = self._neural_network.predict(network_input)
        return np.reshape(output, -1)
