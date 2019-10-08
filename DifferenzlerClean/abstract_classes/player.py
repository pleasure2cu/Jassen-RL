import abc
from typing import Any

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
    def finish_round(self, prediction: int, made_points: int, train: bool) -> Any:  # returns statistics
        pass
