import numpy as np


class GameState:

    blies_history: np.ndarray
    current_blie_index: int
    points_made: np.ndarray
    predictions: np.ndarray

    def __init__(self):
        self.blies_history = np.ones((9, 9)) * -1
        self.current_blie_index = 0
        self.points_made = np.zeros(4)

    def add_card(self, card: np.ndarray, player_index: int):
        self.blies_history[self.current_blie_index][2*player_index: 2*(player_index+1)] = card

    def set_starting_player_of_blie(self, player_index: int):
        self.blies_history[self.current_blie_index][-1] = player_index
