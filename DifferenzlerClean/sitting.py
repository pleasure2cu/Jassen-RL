import datetime
import random
from operator import itemgetter
from typing import Any, Tuple, List, Dict

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
        print("The sitting received a new set of players with length {}".format(len(players)))

    def play_cards(self, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._players) % 4:
            print("The number of players in the sitting has to be a multiple of 4. But is", len(self._players))
            exit()
        nbr_of_tables = len(self._players) // 4
        states = [GameState() for _ in range(nbr_of_tables)]
        if shuffle:
            tmp = list(enumerate(self._players))
            random.shuffle(tmp)
            shuffle_indices, self._players = zip(*tmp)
        for i in range(nbr_of_tables):
            self._deal_cards(player_offset=4*i)
            states[i].predictions = np.array([player.make_prediction() for player in self._players[i*4: (i+1)*4]])

        for i in range(nbr_of_tables):
            assert np.array_equal(
                np.sum([player._hand_vector for player in self._players[4*i: 4*(i+1)]], axis=0),
                np.ones(36)
            ), np.sum([player._hand_vector for player in self._players[4*i: 4*(i+1)]], axis=0)

        player_indices = np.zeros(nbr_of_tables, dtype=int)
        for blie_index in range(9):
            for state_i in range(nbr_of_tables):
                states[state_i].current_blie_index = blie_index
                states[state_i].set_starting_player_of_blie(player_indices[state_i])
            table_suits = -1 * np.ones(nbr_of_tables, dtype=int)
            for inside_blie_turn_counter in range(4):
                played_cards = np.empty((nbr_of_tables, 2))
                turn_player_from_each_table = [
                    (
                        self._players[4*table_index + player_index].strategy_model,
                        4*table_index + player_index,
                        self._players[4 * table_index + player_index].form_nn_input_tensors(
                            states[table_index],
                            table_suits[table_index]
                        )
                    )
                    for table_index, player_index in enumerate(player_indices)
                ]

                # the goal of this part is to group the triples by the first entry
                unique_nets = set(map(itemgetter(0), turn_player_from_each_table))
                triplets_aggregated_by_net: Dict[keras.Model, List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]] = \
                    dict(zip(unique_nets, map(lambda _: [], unique_nets)))
                for t in turn_player_from_each_table:
                    triplets_aggregated_by_net[t[0]].append(t[1:])

                for net, indices_and_tensors_list in triplets_aggregated_by_net.items():
                    rnn_tensors, aux_tensors = list(zip(*map(itemgetter(1), indices_and_tensors_list)))
                    nn_input = [
                        np.concatenate(rnn_tensors, axis=0),
                        np.concatenate(aux_tensors, axis=0)
                    ]
                    tmp = datetime.datetime.now()
                    q_values = net.predict(nn_input).reshape(-1) if len(nn_input[1]) != 0 else []
                    RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp

                    offset = 0
                    for player_i, player_tensors in indices_and_tensors_list:
                        nbr_of_inputs = len(player_tensors[1])
                        played_cards[player_i // 4] = \
                            self._players[player_i].get_action(q_values[offset: offset + nbr_of_inputs])
                        offset += nbr_of_inputs

                if inside_blie_turn_counter == 0:
                    for table_i in range(nbr_of_tables):
                        table_suits[table_i] = int(played_cards[table_i, -1])
                for table_i in range(nbr_of_tables):
                    states[table_i].add_card(played_cards[table_i], player_indices[table_i])

                player_indices = (player_indices + 1) % 4

            tables = [
                np.reshape(states[state_i].blies_history[states[state_i].current_blie_index][:8], (4, 2))
                for state_i in range(nbr_of_tables)
            ]
            winning_index = [
                get_winning_card_index(tables[table_i], player_indices[table_i])
                for table_i in range(nbr_of_tables)
            ]
            points_on_table = [
                get_points_from_table(tables[table_i], blie_index == 8) for table_i in range(nbr_of_tables)
            ]
            for state_i in range(nbr_of_tables):
                states[state_i].points_made[winning_index[state_i]] += points_on_table[state_i]
            player_indices = np.array(winning_index)

        for i in range(nbr_of_tables):
            assert np.sum(states[i].points_made) == 157

        predictions = np.array([states[state_i].predictions for state_i in range(nbr_of_tables)]).reshape(-1)
        points_made = np.array([states[state_i].points_made for state_i in range(nbr_of_tables)]).reshape(-1)
        if shuffle:
            self._players = reverse_shuffle(self._players, shuffle_indices)
            predictions = np.array(reverse_shuffle(predictions, shuffle_indices))
            points_made = np.array(reverse_shuffle(points_made, shuffle_indices))
        return predictions.reshape((-1, 4)), points_made.reshape((-1, 4))

    def play_full_round(self, train: bool, discount: float=0.0, shuffle: bool = True) -> Any:
        predictions, points_made = self.play_cards(shuffle=shuffle)
        diffs = np.absolute(predictions - points_made)
        discount_factors = np.apply_along_axis(get_discount_factors, axis=1, arr=diffs)
        for pred, made, player, disc_factor in zip(
                predictions.reshape(-1), points_made.reshape(-1), self._players, discount_factors.reshape(-1)
        ):
            player.finish_round(pred, made, train, disc_factor * discount)
        over_all_diffs = np.sum(diffs, axis=0)
        return over_all_diffs

    def _deal_cards(self, player_offset: int = 0):
        distribution = np.random.permutation(np.arange(36))
        for i in range(4):
            indices = distribution[i * 9: (i + 1) * 9]
            hand = np.zeros(36)
            hand[indices] = 1
            self._players[i + player_offset].start_round(hand, i)


def get_discount_factors(table_diff: np.ndarray) -> np.ndarray:
    min_indices = np.where(table_diff == np.min(table_diff))[0]
    discount_factors = np.zeros(4)  # so the number of the players at the table
    discount_factors[min_indices] = 1. / len(min_indices)
    return discount_factors
