from typing import List, Tuple

import numpy as np

from PlayerInterlayer import PlayerInterlayer, RnnPlayerInterlayer
from helper_functions import get_winning_card_index, get_points_from_table


class Sitting:
    _players: List[RnnPlayerInterlayer]

    def __init__(self, players: List[RnnPlayerInterlayer]):
        self._players = players

    def play_full_round(self) -> int:
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

        # predictions = predictions // 2

        # assertion
        tmp_hand = np.zeros(36)
        for i in range(4):
            tmp_hand += self._players[i]._player.hand
        assert np.all(tmp_hand == 1), "the cards the players know don't form a full set of cards"

        # play cards
        gone_cards = np.zeros(36)
        blie_history: List[Tuple[np.ndarray, int]] = []
        for blie_index in range(9):
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
        round_winner_index = np.argmin(absolute_diff)
        for i in range(len(self._players)):
            # rewards for the strategy are chosen s.t. the expected value is 0
            if i == round_winner_index:
                self._players[i].end_round(points_made[i], 0.75)
            else:
                self._players[i].end_round(points_made[i], -0.25)

        assert np.sum(points_made) == 157
        assert not np.any(points_made < 0)

        # just for stat
        return int(np.sum(absolute_diff))
