import datetime
import random
from typing import Any, Tuple, List

import keras
import numpy as np

from abstract_classes.player import DifferenzlerPlayer
from abstract_classes.sitting import Sitting
from helpers import get_points_from_table, get_winning_card_index
from player import RnnPlayer
from state import GameState


def shuffle_list(list_to_be_shuffled: List, start_i: int, end_i: int) -> Tuple[List, List[int]]:
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

    def play_cards(self, nbr_of_parallel_rounds: int = 1, strategy_model: keras.Model=None) -> Tuple[np.ndarray, np.ndarray]:
        assert nbr_of_parallel_rounds == 1 or strategy_model is not None
        states = [GameState() for _ in range(nbr_of_parallel_rounds)]
        shuffle_indices: List[int] = [None] * 4 * nbr_of_parallel_rounds
        for i in range(0, 4 * nbr_of_parallel_rounds, 4):
            self._players[i: i+4], shuffle_indices[i: i+4] = shuffle_list(self._players, i, i+4)
        for offset in range(0, nbr_of_parallel_rounds * 4, 4):
            self._deal_cards(player_offset=offset)

        for i in range(nbr_of_parallel_rounds):
            states[i].predictions = np.array([player.make_prediction() for player in self._players[i*4: (i+1)*4]])

        player_index = np.zeros(nbr_of_parallel_rounds, dtype=int)
        for blie_index in range(9):
            for state_i in range(nbr_of_parallel_rounds):
                states[state_i].current_blie_index = blie_index
                states[state_i].set_starting_player_of_blie(player_index[state_i])
            table_suit = -1 * np.ones(nbr_of_parallel_rounds, dtype=int)
            for offset in range(4):
                if nbr_of_parallel_rounds > 1:
                    inputs = [
                        self._players[i*4+player_index[i]].form_nn_input_tensors(states[i], table_suit[i])
                        for i in range(nbr_of_parallel_rounds)
                    ]
                    filtered_inputs = filter(lambda x: len(x[0]), inputs)
                    rnn_and_aux_aggregated = list(zip(*filtered_inputs))
                    if len(rnn_and_aux_aggregated) != 0:
                        nn_input = [
                            np.concatenate(rnn_and_aux_aggregated[0], axis=0),
                            np.concatenate(rnn_and_aux_aggregated[1], axis=0)
                        ]
                        tmp = datetime.datetime.now()
                        q_values = strategy_model.predict(nn_input).reshape(-1) if len(nn_input[1]) != 0 else []
                        RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
                    else:
                        q_values = []
                    q_values_per_player = [len(aux) for rnn, aux in inputs]
                    q_offsets = [0] + [sum(q_values_per_player[:i]) for i in range(1, len(q_values_per_player) + 1)]
                    played_cards = []
                    for i in range(nbr_of_parallel_rounds):
                        played_cards.append(
                            self._players[i*4+player_index[i]].get_action(q_values[q_offsets[i]: q_offsets[i+1]])
                        )
                else:
                    played_cards = [self._players[player_index[0]].play_card(states[0], table_suit[0])]

                for parallel_i in range(nbr_of_parallel_rounds):
                    if table_suit[parallel_i] < 0:
                        table_suit[parallel_i] = int(played_cards[parallel_i][-1])
                    states[parallel_i].add_card(played_cards[parallel_i], player_index[parallel_i])

                player_index = (player_index + 1) % 4

            tables = [np.reshape(states[state_i].blies_history[states[state_i].current_blie_index][:8], (4, 2)) for state_i in range(nbr_of_parallel_rounds)]
            winning_index = [get_winning_card_index(tables[table_i], player_index[table_i]) for table_i in range(nbr_of_parallel_rounds)]
            points_on_table = [get_points_from_table(tables[table_i], blie_index == 8) for table_i in range(nbr_of_parallel_rounds)]
            for state_i in range(nbr_of_parallel_rounds):
                states[state_i].points_made[winning_index[state_i]] += points_on_table[state_i]
            player_index = np.array(winning_index)
        for i in range(nbr_of_parallel_rounds):
            assert np.sum(states[i].points_made) == 157
        self._players = reverse_shuffle(self._players, shuffle_indices)
        predictions = np.array([states[state_i].predictions for state_i in range(nbr_of_parallel_rounds)]).reshape(-1)
        predictions = np.array(reverse_shuffle(predictions, shuffle_indices)).reshape((-1, 4))
        points_made = np.array([states[state_i].points_made for state_i in range(nbr_of_parallel_rounds)]).reshape(-1)
        points_made = np.array(reverse_shuffle(points_made, shuffle_indices)).reshape((-1, 4))
        return predictions, points_made

    def play_full_round(
            self, train: bool, nbr_of_parallel_rounds: int=1, strategy_model: keras.Model=None, discount: float=0.0
    ) -> Any:
        assert nbr_of_parallel_rounds == 1 or strategy_model is not None
        predictions, points_made = self.play_cards(
            nbr_of_parallel_rounds=nbr_of_parallel_rounds,
            strategy_model=strategy_model
        )
        diffs = np.absolute(predictions - points_made)
        for i in range(nbr_of_parallel_rounds):  # TODO: check if optimizable
            diffs_here = diffs[i]
            indices_of_winners = np.where(diffs_here == np.min(diffs_here))[0]
            winners = [self._players[i*4+index] for index in indices_of_winners]
            for prediction, made, player in zip(predictions[i], points_made[i], self._players[i*4: (i+1)*4]):
                player.finish_round(
                    prediction, made, train, discount=discount / len(winners) if player in winners else 0.0
                )
        over_all_diffs = np.sum(np.absolute(predictions - points_made), axis=0)
        return over_all_diffs

    def _deal_cards(self, player_offset: int = 0):
        distribution = np.random.permutation(np.arange(36))
        for i in range(4):
            indices = distribution[i * 9: (i + 1) * 9]
            hand = np.zeros(36)
            hand[indices] = 1
            self._players[i + player_offset].start_round(hand, i)
