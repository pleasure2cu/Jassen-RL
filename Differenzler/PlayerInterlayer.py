from typing import Union, List, Tuple, Callable

import numpy as np

from Player import Player, RnnPlayer
from Sample import RnnNetInput, RnnState, RnnSample, MultiPredictionSample
from helper_functions import TNRepresentation, prediction_state_37_booster, state_action_83_booster, rnn_sample_booster


class PlayerInterlayer:
    _player: Player
    _prediction_log: np.array
    _strategy_log: np.ndarray
    _round_index: int
    _absolute_position_at_table: np.array  # 4 entries vector
    pred_y_func: Callable
    strat_y_func: Callable

    def __init__(self, player: Player, size_of_one_strat_net_input: int, pred_y_func: Callable, strat_y_func: Callable):
        self._player = player
        self._strategy_log = np.empty((9, size_of_one_strat_net_input))
        self._absolute_position_at_table = None
        self.pred_y_func = pred_y_func
        self.strat_y_func = strat_y_func

    def set_absolute_position(self, position: int):
        assert 0 <= position < 4, position
        tmp = np.zeros(4)
        tmp[position] = 1
        self._absolute_position_at_table = tmp

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

    def play_card(self, table_cards: np.ndarray, index_of_first_card: int, gone_cards: np.array,
                  diff: int) -> TNRepresentation:
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
        # add samples to the replay memory for predictions
        prediction_state_vectors = prediction_state_37_booster(self._prediction_log)
        prediction_samples = [(sv, prediction_reward) for sv in prediction_state_vectors]
        self._player.prediction_network.add_samples(prediction_samples)

        # add samples to the replay memory for strategy
        strategy_state_action_vectors = []
        for sav in self._strategy_log:
            strategy_state_action_vectors += state_action_83_booster(sav)
        strategy_samples = [(sav, strategy_reward) for sav in strategy_state_action_vectors]
        self._player.strategy_network.add_samples(strategy_samples)

        # assertions for testing time
        for i in range(9):
            assert np.sum(self._strategy_log[i][8:44]) == 9 - i, "hand cards aren't kept track of correctly"
            assert 4 * (i + 1) > np.sum(
                self._strategy_log[i][44:80]) >= 4 * i, "gone cards aren't kept track of correctly"
            assert i == 0 or self._strategy_log[i][80] <= self._strategy_log[i - 1][80]
            for j in range(i + 1, 9):
                assert not np.array_equal(self._strategy_log[i][-2:], self._strategy_log[j][-2:])

        self._strategy_log = np.empty(self._strategy_log.shape)


class RnnPlayerInterlayer(PlayerInterlayer):
    _player: RnnPlayer
    _strategy_log: List[RnnNetInput]

    def __init__(self, player: Player, pred_y_func: Callable, strat_y_func: Callable):
        super().__init__(player, 1, pred_y_func, strat_y_func)
        self._strategy_log = []

    def play_card(self, table_cards: np.ndarray, index_of_first_card: int, gone_cards: np.array, diff: int,
                  blie_history: List[Tuple[np.ndarray, int]]) -> TNRepresentation:
        """
        plays a card according to the player object
        :param table_cards: table in the two-numbers-representation, in the absolute "coordinate system"
        :param index_of_first_card: index of the first card that has been played inside the table
        :param gone_cards: 36 entry vector with the gone cards
        :param diff: predicted points minus the points made so far
        :param blie_history: list of tuples where the first entry is the table of that round and the second the index
                    of the player that started that blie
        :return: card we want to play in the two-numbers-representation
        """
        assert self._absolute_position_at_table is not None
        # put together the rnn input
        rnn_input_vector = np.zeros((len(blie_history) + 1) * 9)
        for i in range(len(blie_history)):
            rnn_input_vector[i * 9: i * 9 + 8] = np.reshape(blie_history[i][0], -1)
            rnn_input_vector[i * 9 + 8] = blie_history[i][1]
        rnn_input_vector[-9: -1] = np.reshape(table_cards, -1)
        rnn_input_vector[-1] = index_of_first_card

        # put together the dense input (position(4), table(8), hand(36), gone cards(36), diff(1))
        dense_input_vector = np.concatenate((
            self._absolute_position_at_table,
            np.reshape(table_cards, -1),
            self._player.hand,
            gone_cards,
            np.array([diff])
        ))

        # produce the input to the strategy network
        state = RnnState(np.reshape(rnn_input_vector, (-1, 9)), dense_input_vector)

        # get the action an keep it logged
        net_input, action = self._player.play_card(state, int(table_cards[index_of_first_card][1]))
        self._strategy_log.append(net_input)
        self._round_index += 1
        return action

    def end_round(self, prediction_reward: Union[int, float], strategy_reward: Union[int, float]):
        if not self._player.prediction_network._can_train:
            self._strategy_log = []
            return
        self._player.prediction_network.add_samples([
            (self._prediction_log, prediction_reward)
        ])
        self._player.strategy_network.add_samples([
            RnnSample(log_entry.rnn_input, log_entry.aux_input, strategy_reward) for log_entry in self._strategy_log
        ])
        self._strategy_log = []


class RnnMultiPlayerInterlayer(RnnPlayerInterlayer):
    _prediction_log_prediction: int

    def make_prediction(self, relative_position_at_table: int) -> int:
        prediction = super(RnnMultiPlayerInterlayer, self).make_prediction(relative_position_at_table)
        self._prediction_log_prediction = prediction
        return prediction
    
    def end_round(self, prediction_reward: Union[int, float], strategy_reward: Union[int, float]):
        self._player.prediction_network.add_samples([
            MultiPredictionSample(s, self._prediction_log_prediction, prediction_reward) for s in
            prediction_state_37_booster(self._prediction_log)
        ])
        sample_seeds = [
            RnnSample(log_entry.rnn_input, log_entry.aux_input, strategy_reward) for log_entry in self._strategy_log
        ]
        augmented_samples = []
        for sample_seed in sample_seeds:
            augmented_samples += rnn_sample_booster(sample_seed)
        self._player.strategy_network.add_samples(augmented_samples)
        self._strategy_log = []

