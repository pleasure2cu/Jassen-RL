from typing import Tuple, List

from Network import PredictionNetwork, StrategyNetwork, RnnStrategyNetwork
import numpy as np

from Sample import RnnState, RnnNetInput
from helper_functions import get_all_possible_actions, TNRepresentation


class Player:
    prediction_network: PredictionNetwork
    strategy_network: StrategyNetwork
    _prediction_exp: float
    _strategy_exp: float
    hand: np.array  # vector of 36 entries
    
    def __init__(self, prediction_network: PredictionNetwork, strategy_network: StrategyNetwork,
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

    def play_card(self, state: np.array, played_index: int) -> Tuple[np.array, TNRepresentation]:
        """
        Takes state and the suit index and plays a card
        :param state: array with the state
        :param played_index: index of the first played suit (-1 if no card has been played this round)
        :return: Tuple:
                 1. entry is the whole input for the neural network of the finally chosen action.
                 2. entry is the chosen action
        """
        possible_actions = get_all_possible_actions(self.hand, played_index)
        if np.random.binomial(1, self._strategy_exp):
            action_index = np.random.randint(len(possible_actions))
        else:
            net_inputs: np.ndarray = np.concatenate(
                [np.tile(state, (len(possible_actions), 1)), possible_actions],
                axis=1
            )
            assert len(net_inputs) == len(possible_actions), "there isn't an input for each possible action"
            q_values: np.array = self.strategy_network.evaluate(net_inputs)
            assert len(q_values) == len(net_inputs), "there aren't as many q-values as there were inputs"
            action_index = np.argmax(q_values)
        action = possible_actions[action_index]
        self.hand[action[0] + 9 * action[1]] = 0
        return np.concatenate((state, possible_actions[action_index])), possible_actions[action_index]


class RnnPlayer(Player):
    strategy_network: RnnStrategyNetwork

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


