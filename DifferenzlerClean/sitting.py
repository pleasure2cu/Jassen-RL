import random
from typing import Any, Tuple, List

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


def shuffle_list(list_to_be_shuffled: List, start_i: int, end_i: int):
    shuffle_indices = list(range(start_i, end_i))
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

    def play_cards(self, nbr_of_parallel_rounds: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        states = [GameState() for _ in range(nbr_of_parallel_rounds)]
        shuffle_indices = [100] * 4 * nbr_of_parallel_rounds
        for i in range(0, 4 * nbr_of_parallel_rounds, 4):
            self._players[i: i+4], shuffle_indices[i: i+4] = shuffle_list(self._players, i, i+4)
        for offset in range(0, nbr_of_parallel_rounds * 4, 4):
            self._deal_cards(player_offset=offset)

        for i in range(nbr_of_parallel_rounds):
            states[i].predictions = np.array([player.make_prediction() for player in self._players[i*4: (i+1)*4]])

        player_index = np.zeros(nbr_of_parallel_rounds)
        for blie_index in range(9):
            for state_i in range(nbr_of_parallel_rounds):
                states[state_i].current_blie_index = blie_index
                states[state_i].set_starting_player_of_blie(player_index[state_i])
            table_suit = -1 * np.ones(nbr_of_parallel_rounds)
            for offset in range(4):
                if nbr_of_parallel_rounds > 1:
                    inputs: List[Tuple[np.ndarray, np.ndarray]] = [
                        self._players[int(i*4+player_index[i])].form_nn_input_tensors(states[i], int(table_suit[i]))
                        for i in range(nbr_of_parallel_rounds)
                    ]  # get the individual inputs to the strat network
                    rnn_and_aux_aggregated = list(zip(*inputs))
                    nn_input = [
                        np.concatenate(rnn_and_aux_aggregated[0], axis=0),
                        np.concatenate(rnn_and_aux_aggregated[1], axis=0)
                    ]
                    q_values = self._players[0]._strategy_model.predict(nn_input).reshape(-1)
                    nbr_of_q_values = [len(aux) for rnn, aux in inputs]
                    q_values_offsets = [0] + [sum(nbr_of_q_values[:i]) for i in range(1, len(nbr_of_q_values) + 1)]
                    played_cards = []
                    for i in range(nbr_of_parallel_rounds):
                        played_cards.append(
                            self._players[int(i*4+player_index[i])].get_action(q_values[q_values_offsets[i]: q_values_offsets[i+1]])
                        )
                else:
                    played_cards = [self._players[int(player_index[0])].play_card(states[0], table_suit)]

                for parallel_i in range(nbr_of_parallel_rounds):
                    if table_suit[parallel_i] < 0:
                        table_suit[parallel_i] = played_cards[parallel_i][-1]
                    states[parallel_i].add_card(played_cards[parallel_i], int(player_index[parallel_i]))

                player_index = (player_index + 1) % 4

            tables = [np.reshape(states[state_i].blies_history[states[state_i].current_blie_index][:8], (4, 2)) for state_i in range(nbr_of_parallel_rounds)]
            winning_index = [get_winning_card_index(tables[table_i], int(player_index[table_i])) for table_i in range(nbr_of_parallel_rounds)]
            points_on_table = [get_points_from_table(tables[table_i], blie_index == 8) for table_i in range(nbr_of_parallel_rounds)]
            for state_i in range(nbr_of_parallel_rounds):
                states[state_i].points_made[int(winning_index[state_i])] += points_on_table[state_i]
            player_index = np.array(winning_index)
        for i in range(nbr_of_parallel_rounds):
            assert np.sum(states[i].points_made) == 157
        self._players = reverse_shuffle(self._players, shuffle_indices)
        predictions = np.array([states[state_i].predictions for state_i in range(nbr_of_parallel_rounds)]).reshape(-1)
        predictions = reverse_shuffle(predictions, shuffle_indices)
        predictions = np.array(predictions).reshape((-1, 4))
        points_made = np.array([states[state_i].points_made for state_i in range(nbr_of_parallel_rounds)]).reshape(-1)
        points_made = reverse_shuffle(points_made, shuffle_indices)
        points_made = np.array(points_made).reshape((-1, 4))
        return predictions, points_made

    def play_full_round(self, train: bool, nbr_of_parallel_rounds: int = 1) -> Any:
        predictions, points_made = self.play_cards(nbr_of_parallel_rounds=nbr_of_parallel_rounds)
        total_pred_loss = 0.
        total_strat_loss = 0.
        for i in range(nbr_of_parallel_rounds):
            for prediction, made, player in zip(predictions[i], points_made[i], self._players[i*4: (i+1)*4]):
                p_loss, s_loss = player.finish_round(prediction, made, train)
                total_pred_loss += p_loss
                total_strat_loss += s_loss
        diffs = np.sum(np.absolute(predictions - points_made), axis=0)
        return total_pred_loss, total_strat_loss, diffs

    def _deal_cards(self, player_offset: int = 0):
        distribution = np.random.permutation(np.arange(36))
        for i in range(4):
            indices = distribution[i * 9: (i + 1) * 9]
            hand = np.zeros(36)
            hand[indices] = 1
            self._players[i + player_offset].start_round(hand, i)
