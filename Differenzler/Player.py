from typing import Tuple, List

from Network import PredictionNetwork, RnnStrategyNetwork
import numpy as np

from Sample import RnnState, RnnNetInput
from helper_functions import get_all_possible_actions


class RnnPlayer:
    prediction_network: PredictionNetwork
    strategy_network: RnnStrategyNetwork
    _prediction_exp: float
    _strategy_exp: float
    hand: np.array

    def __init__(self, prediction_network: PredictionNetwork, strategy_network: RnnStrategyNetwork,
                 prediction_exp: float, strategy_exp: float):
        self.prediction_network = prediction_network
        self.strategy_network = strategy_network
        self._prediction_exp = prediction_exp
        self._strategy_exp = strategy_exp

    def receive_hand(self, hand_vector: np.array):
        """
        :param hand_vector: vector with 36 entries
        :return: None
        """
        assert hand_vector.shape == (36,), "is " + str(hand_vector.shape)
        assert np.sum(hand_vector) == 9, hand_vector
        for entry in hand_vector:
            assert entry in [0, 1]
        self.hand = hand_vector

    def make_prediction(self, position_at_table: int) -> Tuple[int, np.array]:
        """
        :return: The points the network thinks this hand will yield
        """
        assert position_at_table in [0, 1, 2, 3], "is " + str(position_at_table)
        state = np.concatenate([
            self.hand,
            np.array([position_at_table])
        ])
        if np.random.binomial(1, self._prediction_exp):
            return np.random.randint(158), state
        return self.prediction_network.evaluate(state), state

    def play_card(self, state: RnnState, suit: int) -> Tuple[RnnNetInput, np.array]:
        """
        :param state: self-describing
        :param suit: in [-1, 4]
        :return: Tuple:
                1. the state-action pair to add to the log
                2. the action chosen
        """
        assert suit in range(-1, 4), suit
        possible_actions = get_all_possible_actions(self.hand, suit)
        n = len(possible_actions)
        rnn_part = np.tile(state.rnn_part, (n, 1, 1))
        aux_part_left_block = np.tile(state.aux_part, (n, 1))
        aux_part = np.concatenate([aux_part_left_block, possible_actions], axis=1)
        if np.random.binomial(1, self._strategy_exp):
            action_index = np.random.randint(n)
        else:
            q_values = self.strategy_network.evaluate([rnn_part, aux_part])
            action_index = np.argmax(q_values)
        action = possible_actions[action_index]
        self.hand[action[0] + 9 * action[1]] = 0
        return RnnNetInput(rnn_part[action_index], aux_part[action_index]), action


