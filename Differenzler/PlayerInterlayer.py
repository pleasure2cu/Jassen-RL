from typing import Union

import numpy as np

from Player import Player
from helper_functions import TNRepresentation


class PlayerInterlayer:
    _player: Player
    _prediction_log: np.array
    _strategy_log: np.ndarray
    _round_index: int

    def __init__(self, player: Player, size_of_one_strat_net_input: int):
        self._player = player
        self._strategy_log = np.empty((9, size_of_one_strat_net_input))

    def receive_hand(self, hand: np.array):
        self._round_index = 0
        self._player.receive_hand(hand)

    def make_prediction(self, position_at_table: int) -> int:
        prediction, state_vector = self._player.make_prediction(position_at_table)
        assert state_vector.size == 37
        self._prediction_log = state_vector
        assert np.sum(self._prediction_log[:36]) == 9
        assert self._prediction_log[-1] in [0, 1, 2, 3]
        return prediction

    def play_card(self, table_cards: np.ndarray, index_of_first_card: int, gone_cards: np.array, diff: int) \
            -> TNRepresentation:
        """
        plays a card according to the player object
        :param table_cards: table in the two-numbers-representation, in the absolute "coordinate system"
        :param index_of_first_card: index of the first card that has been played inside the table
        :param gone_cards: 36 entry vector with the gone cards
        :param diff: predicted points minus the points made so far
        :return: card we want to play in the two-numbers-representation
        """
        # change the "coordinate system" of the table
        relative_table = np.roll(table_cards, len(table_cards) - index_of_first_card, axis=0)
        assert not relative_table[0][0] == -1 or relative_table[0][0] == relative_table[1][0]
        assert not relative_table[1][0] == -1 or relative_table[1][0] == relative_table[2][0]
        assert not relative_table[2][0] == -1 or relative_table[2][0] == relative_table[3][0]
        # put together the state
        # the state vector is: table (-1 where nothing), hand, gone cards, diff
        assert np.sum(self._player.hand) == 9 - self._round_index, "hand doesn't have the correct amount of cards"
        state = np.concatenate((
            np.reshape(relative_table, -1),
            self._player.hand,
            gone_cards,
            np.array([diff])
        ))
        # ask the player for the action
        net_input, action = self._player.play_card(state, index_of_first_card)
        assert gone_cards[action[0] + 9 * action[1]] == 0, "the played card is actually already gone"
        # log the network input
        self._strategy_log[self._round_index] = net_input
        self._round_index += 1
        # return the action
        return action

    def end_round(self, prediction_reward: Union[int, float], strategy_reward: Union[int, float]):
        """
        Does bookkeeping (adding data points to the ReplayMemories, resetting log)
        :param prediction_reward:
        :param strategy_reward:
        :return:
        """
        self._player.prediction_network.add_samples([(self._prediction_log, prediction_reward)])
        mem_entries = map(lambda x: (x, strategy_reward), self._strategy_log)
        # assertions for testing time
        for i in range(9):
            assert np.sum(self._strategy_log[i][8:44]) == 9 - i, "hand cards aren't kept track of correctly"
            assert 4 * (i + 1) > np.sum(self._strategy_log[i][44:80]) >= 4 * i, "gone cards aren't kept track of correctly"
            assert i == 0 or self._strategy_log[i][80] <= self._strategy_log[i-1][80]
            for j in range(i + 1, 9):
                assert not np.array_equal(self._strategy_log[i][-2:], self._strategy_log[j][-2:])
        self._player.strategy_network.add_samples(list(mem_entries))
        self._strategy_log = np.empty(self._strategy_log.shape)
