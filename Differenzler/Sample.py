from typing import Union

import numpy as np


class RnnSample:
    rnn_input: np.ndarray  # 9 entries (table in two-numbers rep. and index of player that started blie) repeated
    aux_input: np.array  # index_of_player(4), table(8), hand(36), gone_cards(36), diff(1), action(2)
    y: Union[int, float]

    def __init__(self, rnn_input: np.ndarray, aux_input: np.array, y: Union[int, float]):
        assert len(rnn_input.shape) == 2
        assert len(aux_input.shape) == 1
        self.rnn_input = rnn_input
        self.aux_input = aux_input
        self.y = y

    def number_of_time_steps(self) -> int:
        return len(self.rnn_input)


class RnnNetInput:
    # this is the exact same class as 'RnnSample' but without the y. Generally, every 'RnnSample' object will have a
    # 'RnnNetInput' predecessor
    rnn_input: np.ndarray
    aux_input: np.array

    def __init__(self, rnn_input: np.ndarray, aux_input: np.array):
        assert len(rnn_input.shape) == 2
        assert len(aux_input.shape) == 1
        self.rnn_input = rnn_input
        self.aux_input = aux_input


class RnnState:
    # note that the data in this class not only differs in the 'y' from the data in an 'RnnSample' object, but the
    # 'aux_part' here doesn't contain information about the action. While 'aux_input' contains the action information

    rnn_part: np.ndarray
    aux_part: np.array

    def __init__(self, rnn_part: np.ndarray, aux_part: np.array):
        assert len(rnn_part.shape) == 2
        assert len(aux_part.shape) == 1
        self.rnn_part = rnn_part
        self.aux_part = aux_part
