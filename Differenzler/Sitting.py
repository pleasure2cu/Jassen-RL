from typing import List, Tuple, Callable

import numpy as np

from PlayerInterlayer import PlayerInterlayer, RnnPlayerInterlayer
from helper_functions import get_winning_card_index, get_points_from_table


class Sitting:
    _players: List[RnnPlayerInterlayer]
    _debugging: bool

    def __init__(self, players: List[RnnPlayerInterlayer], debugging: bool):
        self._players = players
        self._debugging = debugging

    def play_full_round(self) -> int:
        np.random.seed(1)
        # distribute the cards to the players
        distribution = np.random.permutation(np.arange(36))
        for i in range(4):
            indices = distribution[i * 9: (i + 1) * 9]
            hand = np.zeros(36)
            hand[indices] = 1
            self._players[i].receive_hand(hand)

        # gather the predictions
        player_index = np.random.randint(len(self._players))
        predictions = np.zeros(4)
        points_made = np.zeros(4)
        for i in range(4):
            predictions[i] = self._players[i].make_prediction((i - player_index) % 4)

        # assertion
        tmp_hand = np.zeros(36)
        for i in range(4):
            tmp_hand += self._players[i]._player.hand
        assert np.all(tmp_hand == 1), "the cards the players know don't form a full set of cards"

        # play cards
        gone_cards = np.zeros(36)
        blie_history: List[Tuple[np.ndarray, int]] = []
        for blie_index in range(9):
            if self._debugging:
                assert np.sum(gone_cards) == 4 * blie_index
                assert not np.any(gone_cards < 0), "in the gone cards vector there are negative entries"
                assert not np.any(gone_cards > 1), "in the gone cards vector there are entries > 1"
            table = np.ones((4, 2)) * -1
            table_suit = -1
            for i in range(4):
                assert np.sum(table == -1) == (4 - i) * 2
                played_card = self._players[player_index] \
                    .play_card(table, (player_index - i) % 4, gone_cards,
                               predictions[player_index] - points_made[player_index], blie_history)
                gone_cards[played_card[0] + played_card[1] * 9] = 1
                if table_suit < 0:
                    table_suit = played_card[1]
                table[player_index] = played_card
                player_index = (player_index + 1) % 4
            assert not np.any(table == -1), "the table contains entries that are still -1"
            # add this table to the blie history
            blie_history.append((table, player_index))
            # look who won the hand
            winning_index = get_winning_card_index(table, player_index)
            # look how many points are on the table
            points_on_table = get_points_from_table(table, blie_index == 8)
            # update that player's points
            points_made[winning_index] += points_on_table
            # set the index to that player
            player_index = winning_index

        # compute rewards and give them
        absolute_diff = np.absolute(predictions - points_made)
        # round_winner_index = np.argmin(absolute_diff)
        for i, player in enumerate(self._players):
            pred_y = player.pred_y_func(points_made[i])
            strat_y = player.strat_y_func(predictions[i], points_made[i])
            self._players[i].end_round(pred_y, strat_y)

        if self._debugging:
            # a whole heap of assertions
            added_samples = [sub_list[-4:] for sub_list in
                             self._players[0]._player.strategy_network._replay_memory._items]

            # check that time series doesn't change from blie to blie or inside blie
            for i in range(8):
                j = i + 1
                blie_i = added_samples[i]
                blie_j = added_samples[j]
                for k in range(3):
                    assert np.array_equal(blie_i[k].rnn_input[:i], blie_j[k].rnn_input[:i]), \
                        (blie_i[k].rnn_input[:i], blie_j[k].rnn_input[:i])
                    assert np.array_equal(blie_j[k].rnn_input[:i], blie_j[k+1].rnn_input[:i]), \
                        (blie_j[k].rnn_input[:i], blie_j[k+1].rnn_input[:i])

            # check that the absolute position is correct in all samples
            for i in range(9):
                blie = added_samples[i]
                for j in range(4):
                    tmp = np.zeros(4)
                    tmp[j] = 1
                    assert np.array_equal(blie[j].aux_input[:4], tmp), (j, blie[j].aux_input[:4])

            # check that in each blie over all players only 4 unique cards are in the time series (3 cards, once the -1)
            for blie in added_samples:
                numbers = []
                for i in range(4):
                    numbers += [blie[i].rnn_input[-1][j]+blie[i].rnn_input[-1][j+1]*9 for j in range(0, 8, 2)]
                assert np.unique(numbers).size == 4, (np.unique(numbers).size, np.unique(numbers))
                assert -10 in numbers

            # check that the last entry and the table in the sample is the same
            for blie in added_samples:
                for sample in blie:
                    assert np.array_equal(sample.rnn_input[-1][:8], sample.aux_input[4: 12]), \
                        (sample.rnn_input[-1][:8], sample.aux_input[4: 12])

            assert np.sum(points_made) == 157
            assert not np.any(points_made < 0)

        # just for stat
        return int(np.sum(absolute_diff))
