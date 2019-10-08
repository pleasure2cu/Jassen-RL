from typing import Union, List, Tuple, Callable

import numpy as np

from Player import RnnPlayer
from Sample import RnnNetInput, RnnState, RnnSample
from helper_functions import TNRepresentation


def get_table_roll_array(absolute_position) -> np.ndarray:
    output = np.roll(np.arange(8), absolute_position * 2)
    return output


class RnnPlayerInterlayer:
    _player: RnnPlayer
    _strategy_log: List[RnnNetInput]
    _prediction_log: np.array
    _round_index: int
    _absolute_position_at_table: int
    _table_roll_array: np.ndarray
    pred_y_func: Callable
    strat_y_func: Callable

    def __init__(self, player: RnnPlayer, pred_y_func: Callable, strat_y_func: Callable):
        self._player = player
        self._absolute_position_at_table = None
        self.pred_y_func = pred_y_func
        self.strat_y_func = strat_y_func
        self._strategy_log = []

    def set_absolute_position(self, position: int):
        assert 0 <= position < 4, position
        self._absolute_position_at_table = position
        self._table_roll_array = get_table_roll_array(position)
        
    def receive_hand(self, hand: np.array):
        self._round_index = 0
        self._player.receive_hand(hand)

    def make_prediction(self, relative_position_at_table: int) -> int:
        prediction, state_vector = self._player.make_prediction(relative_position_at_table)
        assert state_vector.size == 37
        self._prediction_log = state_vector
        assert np.sum(self._prediction_log[:36]) == 9
        assert self._prediction_log[-1] in [0, 1, 2, 3]
        return prediction

    def play_card(self, table_cards: np.ndarray, index_of_first_card: int, diff: int,
                  blie_history: List[Tuple[np.ndarray, int]]) -> TNRepresentation:
        """
        plays a card according to the player object
        :param table_cards: table in the two-numbers-representation, in the absolute "coordinate system"
        :param index_of_first_card: index of the first card that has been played inside the table
        :param diff: predicted points minus the points made so far
        :param blie_history: list of tuples where the first entry is the table of that round and the second the index
                    of the player that started that blie
        :return: card we want to play in the two-numbers-representation
        """
        assert self._absolute_position_at_table is not None
        # put together the rnn input
        rnn_input_vector = np.ones((len(blie_history) + 1) * 9) * -1
        roll_array = self._table_roll_array
        for i in range(len(blie_history)):
            rnn_input_vector[i * 9: i * 9 + 8] = np.reshape(blie_history[i][0], -1)[roll_array]
            rnn_input_vector[i * 9 + 8] = blie_history[i][1] % 4
        rnn_input_vector[-9: -1] = np.reshape(table_cards, -1)[roll_array]
        rnn_input_vector[-1] = index_of_first_card % 4

        dense_input_vector = np.concatenate((
            np.reshape(table_cards, -1)[roll_array],
            self._player.hand,
            np.array([diff])
        ))

        # produce the input to the strategy network
        state = RnnState(np.reshape(rnn_input_vector, (-1, 9)), dense_input_vector)

        # get the action and keep it logged
        net_input, action = self._player.play_card(state, int(table_cards[index_of_first_card][1]))
        self._strategy_log.append(net_input)
        self._round_index += 1
        return action

    def end_round(self, prediction_reward: Union[int, float], strategy_reward: Union[int, float]):
        if not self._player.prediction_network.can_train:
            self._strategy_log = []
            return
        self._player.prediction_network.add_samples([
            (self._prediction_log, prediction_reward)
        ])
        self._player.strategy_network.add_samples([
            RnnSample(log_entry.rnn_input, log_entry.aux_input, strategy_reward) for log_entry in self._strategy_log
        ])
        self._strategy_log = []
