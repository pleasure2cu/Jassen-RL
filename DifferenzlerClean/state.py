from typing import Optional

import numpy as np

from helpers import get_points_from_table


class GameState:
    blies_history: np.ndarray
    current_blie_index: int
    points_made: np.ndarray
    predictions: np.ndarray
    gone_cards: np.ndarray
    could_follow: np.ndarray  # 4 by 4 matrix, first index is the suit, second the player. A 1 means it could follow
    could_follow_cache_nbr: int  # after how many played cards the member above was populated

    def __init__(self):
        self.blies_history = np.ones((9, 9)) * -1
        self.current_blie_index = 0
        self.points_made = np.zeros(4)
        self.gone_cards = np.zeros(36, dtype=np.bool)
        self.could_follow = np.ones((4, 4), dtype=np.bool)
        self.could_follow_cache_nbr = 0

    def add_card(self, card: np.ndarray, player_index: int):
        self.blies_history[self.current_blie_index][2 * player_index: 2 * (player_index + 1)] = card
        self.gone_cards[int(card[0] + 9 * card[1])] = 1

    def set_starting_player_of_blie(self, player_index: int):
        self.blies_history[self.current_blie_index][-1] = player_index

    def get_could_follow_vector(self, roll_vector: Optional[np.ndarray] = None) -> np.ndarray:
        nbr_of_gone_cards = int(np.sum(self.gone_cards))
        for i in range(self.could_follow_cache_nbr, nbr_of_gone_cards):
            blie_index = i // 4
            player_index = i % 4
            blie_starter_index = int(self.blies_history[blie_index, -1])
            blie_color = int(self.blies_history[blie_index, 2 * blie_starter_index + 1])
            player_color = int(self.blies_history[blie_index, 2 * player_index + 1])
            if player_color != 0 and blie_color != player_color:
                self.could_follow[blie_color, player_index] = 0
        self.could_follow_cache_nbr = nbr_of_gone_cards
        roll_vector = roll_vector if not (roll_vector is None) else np.arange(4)
        return self.could_follow[roll_vector].reshape(-1)

    def get_points_on_table(self) -> int:
        blie_so_far = self.blies_history[self.current_blie_index, :8]
        tmp: np.ndarray = blie_so_far[blie_so_far >= 0]
        return get_points_from_table(tmp.reshape((-1, 2)), np.sum(self.gone_cards) >= 34)
