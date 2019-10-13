import abc
from typing import Any, Tuple, Union

import numpy as np

from state import GameState


class DifferenzlerPlayer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def start_round(self, hand_vector: np.ndarray, table_position: int):
        pass

    @abc.abstractmethod
    def make_prediction(self) -> int:
        pass

    @abc.abstractmethod
    def play_card(self, state: GameState, suit: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def form_nn_input_tensors(self, state: GameState, suit: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def get_action(self, q_values: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def finish_round(self, prediction: int, made_points: int, train: bool, discount: Union[int, float]=0.0) -> Any:
        # returns statistics
        pass
