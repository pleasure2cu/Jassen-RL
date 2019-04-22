from typing import List

from Memory import ReplayMemory, RnnReplayMemory, MultiReplayMemory
import keras
import numpy as np

from Sample import MultiPredictionSample
from helper_functions import turn_rnn_samples_into_batch


class Network:
    _replay_memory: ReplayMemory
    _neural_network: keras
    _batch_size: int

    def __init__(self, neural_network: keras, memory: ReplayMemory, batch_size: int):
        self._neural_network = neural_network
        self._replay_memory = memory
        self._batch_size = batch_size

    def evaluate(self, network_input: np.ndarray) -> np.array:
        pass

    def add_samples(self, samples: list):
        self._replay_memory.add_samples(samples)

    def train(self) -> float:
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


class MultiPredictionNetwork(PredictionNetwork):
    def __init__(self, neural_network: keras, memory: ReplayMemory, batch_size: int):
        assert type(memory) == MultiReplayMemory, type(memory)
        super().__init__(neural_network, memory, batch_size)

    def evaluate(self, network_input: np.ndarray):
        output_activations = self._neural_network.predict(np.reshape(network_input, (1, -1)))
        skew = 158 // output_activations.size
        return np.argmax(np.reshape(output_activations, -1)) * skew

    def train(self) -> float:
        samples: List[MultiPredictionSample] = self._replay_memory.draw_samples(self._batch_size)
        n = len(samples)
        if n == 0:
            return -1
        # get the current activations
        x_batch = np.zeros((n, samples[0].state_vector.size))
        for i in range(n):
            x_batch[i] = samples[i].state_vector
        y_batch = self._neural_network.predict(x_batch)
        # only change the activation from which we know the 'right answer', given by the sample
        skew = 158 // y_batch[0].size
        for i in range(n):
            y_batch[i][samples[i].guessed_points//skew] = samples[i].reward
        return self._neural_network.train_on_batch(x_batch, y_batch)


class StrategyNetwork(Network):
    def evaluate(self, network_input: np.ndarray) -> np.array:
        """
        :param network_input: has to be in the format needed by the neural network
        :return: q-values
        """
        output = self._neural_network.predict(network_input)
        return np.reshape(output, -1)


class RnnStrategyNetwork(StrategyNetwork):
    _replay_memory: RnnReplayMemory

    def train(self) -> float:
        samples = self._replay_memory.draw_samples(self._batch_size)
        n = len(samples)
        if n == 0:
            return -1
        x_rnn, x_aux, y = turn_rnn_samples_into_batch(samples)
        return self._neural_network.train_on_batch([x_rnn, x_aux], y)

    def evaluate(self, network_input: List[np.ndarray]):
        """
        :param network_input: has two entries, the first one needs to be 3 dimensional (input for RNN part). The second
                part needs to be two dimensional (input for the auxiliary part)
        :return: the returned values as vector (1D)
        """
        assert len(network_input[0].shape) == 3
        assert len(network_input[1].shape) == 2
        output = self._neural_network.predict(network_input)
        return np.reshape(output, -1)
