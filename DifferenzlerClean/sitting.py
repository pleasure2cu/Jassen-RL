import random
from typing import Any, Tuple, List, Iterable

import numpy as np

from abstract_classes.player import DifferenzlerPlayer
from abstract_classes.sitting import Sitting
from state import GameState


trump_points = {0: 20, 1: 14, 2: 11, 3: 4, 4: 3, 5: 10, 6: 0, 7: 0, 8: 0}
color_points = {0: 11, 1: 4, 2: 3, 3: 2, 4: 10, 5: 0, 6: 0, 7: 0, 8: 0}


def get_winning_card_index(table: np.ndarray, first_played_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :return: index of the winning hand
    """
    highest_index = first_played_index
    for i in range(1, 4):
        j = (i + first_played_index) % 4
        if table[highest_index][1] != 0 and table[j][1] == 0:  # trump after non-trump
            highest_index = j
        elif table[highest_index][1] == table[j][1] and table[j][0] < table[highest_index][0]:
            highest_index = j
    return highest_index


def get_points_from_table(table: np.ndarray, last_round: bool) -> int:
    total = 5 if last_round else 0
    for card in table:
        total += color_points[card[0]] if card[1] > 0 else trump_points[card[0]]
    return total


def shuffle_list(list_to_be_shuffled: List):
    shuffle_indices = list(range(len(list_to_be_shuffled)))
    random.shuffle(shuffle_indices)
    output = [list_to_be_shuffled[i] for i in shuffle_indices]
    return output, shuffle_indices


def reverse_shuffle(list_to_be_reversed, shuffle_indices: List[int]):
    assert len(list_to_be_reversed) == len(shuffle_indices), "{}, {}"\
        .format(len(list_to_be_reversed), len(shuffle_indices))
    return [list_to_be_reversed[shuffle_indices.index(i)] for i in range(len(shuffle_indices))]


class DifferenzlerSitting(Sitting):

    _players: List[DifferenzlerPlayer]

    def set_players(self, players: List[DifferenzlerPlayer]):
        self._players = players

    def play_cards(self) -> Tuple[np.ndarray, np.ndarray]:
        state = GameState()
        self._players, shuffle_indices = shuffle_list(self._players)
        self._deal_cards()

        state.predictions = np.array([player.make_prediction() for player in self._players])

        player_index = 0
        for blie_index in range(9):
            state.current_blie_index = blie_index
            state.set_starting_player_of_blie(player_index)
            table_suit = -1
            for i in range(4):
                played_card = self._players[player_index].play_card(state, table_suit)
                if table_suit < 0:
                    table_suit = played_card[-1]
                state.add_card(played_card, player_index)
                player_index = (player_index + 1) % 4

            table = np.reshape(state.blies_history[state.current_blie_index][:8], (4, 2))
            winning_index = get_winning_card_index(table, player_index)
            points_on_table = get_points_from_table(table, blie_index == 8)
            state.points_made[winning_index] += points_on_table
            player_index = winning_index
        assert np.sum(state.points_made) == 157
        self._players = reverse_shuffle(self._players, shuffle_indices)
        predictions = reverse_shuffle(state.predictions, shuffle_indices)
        points_made = reverse_shuffle(state.points_made, shuffle_indices)
        return np.array(predictions), np.array(points_made)

    def play_full_round(self, train: bool) -> Any:
        predictions, points_made = self.play_cards()
        total_pred_loss = 0.
        total_strat_loss = 0.
        for prediction, made, player in zip(predictions, points_made, self._players):
            p_loss, s_loss = player.finish_round(prediction, made, train)
            total_pred_loss += p_loss
            total_strat_loss += s_loss
        return total_pred_loss, total_strat_loss, np.absolute(predictions - points_made)

    def _deal_cards(self):
        distribution = np.random.permutation(np.arange(36))
        for i in range(4):
            indices = distribution[i * 9: (i + 1) * 9]
            hand = np.zeros(36)
            hand[indices] = 1
            self._players[i].start_round(hand, i)
