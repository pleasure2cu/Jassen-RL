import abc
from typing import List, Tuple, Any

import numpy as np

from abstract_classes.player import DifferenzlerPlayer


class Sitting(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def set_players(self, players: List[DifferenzlerPlayer]):
        pass

    @abc.abstractmethod
    def play_cards(self) -> Tuple[np.ndarray, np.ndarray]:  # predictions and made points
        pass

    @abc.abstractmethod
    def play_full_round(self, train: bool) -> Any:  # returns statistics
        pass
