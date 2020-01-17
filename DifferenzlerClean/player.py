import datetime
from itertools import chain
from typing import List, Tuple, Callable, Union, Any

import keras
import numpy as np

from abstract_classes.player import DifferenzlerPlayer
from helpers import which_are_bock
from memory import ReplayMemory, RnnReplayMemory
from sample_boosting import boost_basic_prediction_vector, boost_hand_crafted_strategy_vector
from state import GameState


def translate_vector_to_two_number_representation(hand: np.array) -> np.ndarray:
    card_indices = np.nonzero(hand)[0]
    wanted_representation = [[x % 9, x // 9] for x in card_indices]
    return np.array(wanted_representation)


def get_possible_actions(hand_vector: np.ndarray, first_played_suit: int) -> np.ndarray:
    """
    This function gets all the cards that are allowed to be played according to the rules of the game
    and returns them in the two-numbers representation
    :param hand_vector: 36 entry vector
    :param first_played_suit: in [-1, 4]
    :return: all possible actions in two number representation as np.ndarray (first axis are the different options)
    """
    assert first_played_suit in range(-1, 4), "the value is " + str(first_played_suit)
    if first_played_suit == 0:
        if np.any(hand_vector[1: 9]):
            playable_cards = hand_vector[:9]
        else:
            playable_cards = hand_vector
    elif np.any(hand_vector[first_played_suit * 9: (first_played_suit + 1) * 9]):
        playable_cards = np.zeros(36)
        playable_cards[:9] = hand_vector[:9]
        playable_cards[first_played_suit * 9: (first_played_suit + 1) * 9] = \
            hand_vector[first_played_suit * 9: (first_played_suit + 1) * 9]
    else:
        playable_cards = hand_vector
    return translate_vector_to_two_number_representation(playable_cards)


swaps = [
    [0, 1, 2, 3, -1],
    [0, 1, 3, 2, -1],
    [0, 2, 1, 3, -1],
    [0, 2, 3, 1, -1],
    [0, 3, 1, 2, -1],
    [0, 3, 2, 1, -1],
]


class RnnPlayer(DifferenzlerPlayer):

    # the RNN expects a 2D matrix (time series, 9), where the current table is also included in the time series.
    #   the 9 consists of first 4 cards in two-numbers-representation (-1 if not given) followed by the index of the
    #   person who started the corresponding blie. Every blie should be rolled by -2 * index of the current player
    #   So the direction is "into the stomach of the player"
    # The aux part expects:
    #   - hand vector (36 entry, multiple hot vector)
    #   - relative table of current blie (8 entries vector)
    #   - prediction minus made points of the current player (so a scalar)
    #   - action as two-numbers representation

    _prediction_model: keras.Model
    strategy_model: keras.Model
    _prediction_memory: ReplayMemory
    _strategy_memory: RnnReplayMemory
    _prediction_exp: float
    _strategy_exp: float
    _prediction_pool: List[np.ndarray]
    _strategy_pool: List[Tuple[np.ndarray, np.ndarray]]
    _prediction_y_function: Callable[[int, int], Union[int, float]]  # (angesagt, gemacht) -> |R
    _strategy_y_function: Callable[[int, int], Union[int, float]]  # (angesagt, gemacht) -> |R
    _frozen: bool

    _hand_vector: np.ndarray
    _table_position: int
    _table_roll_indices: np.ndarray
    _batch_size_strat: int

    total_time_spent_in_keras = datetime.timedelta()
    time_spent_training = datetime.timedelta()

    def __init__(
            self,
            prediction_model: keras.Model, strategy_model: keras.Model,
            prediction_memory: ReplayMemory, strategy_memory: RnnReplayMemory,
            prediction_y_function: Callable[[int, int], Union[int, float]],
            strategy_y_function: Callable[[int, int], Union[int, float]],
            prediction_exp: float, strategy_exp: float,
            batch_size_pred:int, batch_size_strat: int,
            frozen: bool = False
    ):
        self._prediction_model = prediction_model
        self.strategy_model = strategy_model
        self._prediction_memory = prediction_memory
        self._strategy_memory = strategy_memory
        self._prediction_exp = prediction_exp
        self._strategy_exp = strategy_exp
        self._prediction_y_function = prediction_y_function
        self._strategy_y_function = strategy_y_function
        self._batch_size_pred = batch_size_pred
        self._batch_size_strat = batch_size_strat
        self._frozen = frozen

    def start_round(self, hand_vector: np.ndarray, table_position: int):
        self._prediction_pool = []
        self._strategy_pool = []
        self._hand_vector = hand_vector
        self._table_position = table_position
        self._table_roll_indices = np.concatenate([np.roll(np.arange(8), -table_position * 2), [8]])

    def make_prediction(self) -> int:
        model_input = np.reshape(np.concatenate([self._hand_vector, [self._table_position]]), (1, -1))
        self._prediction_pool.append(model_input)
        if np.random.binomial(1, self._prediction_exp):
            return np.random.randint(158)
        tmp = datetime.datetime.now()
        prediction = int(self._prediction_model.predict(model_input)[0] + 0.5)
        RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
        return prediction

    _current_possible_actions: np.ndarray
    _current_rnn_state_tensors: np.ndarray
    _current_aux_state_action_tensors: np.ndarray

    def form_nn_input_tensors(self, state: GameState, suit: int) -> Tuple[np.ndarray, np.ndarray]:
        possible_actions = get_possible_actions(self._hand_vector, suit)
        nbr_of_actions = len(possible_actions)
        if np.sum(self._hand_vector) == 1:
            self._current_possible_actions = possible_actions
            return np.array([]), np.array([])
        rnn_state_tensor = np.tile(
            self._get_relative_rnn_input(state),
            (nbr_of_actions, 1, 1)
        )
        relative_table = state.blies_history[state.current_blie_index][self._table_roll_indices[:8]]
        current_difference = [state.predictions[self._table_position] - state.points_made[self._table_position]]
        aux_state_tensor = np.tile(
            np.concatenate([self._hand_vector, relative_table, current_difference]),
            (nbr_of_actions, 1)
        )
        aux_state_action_tensor = np.concatenate([aux_state_tensor, possible_actions], axis=1)
        self._current_possible_actions = possible_actions
        self._current_rnn_state_tensors = rnn_state_tensor
        self._current_aux_state_action_tensors = aux_state_action_tensor
        return rnn_state_tensor, aux_state_action_tensor

    def get_action(self, q_values: np.ndarray) -> np.ndarray:
        if len(q_values) == 0:
            action = self._current_possible_actions[0]
        else:
            if np.random.binomial(1, self._strategy_exp):
                index = np.random.randint(len(q_values))
            else:
                index = np.argmin(q_values)
            self._strategy_pool.append((self._current_rnn_state_tensors[index], self._current_aux_state_action_tensors[index]))
            action = self._current_possible_actions[index]
        self._hand_vector[action[0] + 9 * action[1]] = 0
        return action

    def play_card(self, state: GameState, suit: int) -> np.ndarray:
        rnn_state_tensor, aux_state_action_tensor = self.form_nn_input_tensors(state, suit)
        tmp = datetime.datetime.now()
        if len(aux_state_action_tensor) == 0:
            q_values = []
        else:
            q_values = self.strategy_model.predict([rnn_state_tensor, aux_state_action_tensor])
            print(q_values)
        RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
        return self.get_action(q_values)

    def finish_round(
            self, prediction: int, made_points: int, train: bool, discount: Union[int, float]=0.0
    ):
        assert np.all(self._hand_vector == 0)
        if self._frozen:
            return
        # boost the samples
        boosted_pred_pool = list(chain.from_iterable(
            map(lambda sample: self.boost_color_pred_sample(sample.reshape(-1)), self._prediction_pool)
        ))
        boosted_strat_pool = list(chain.from_iterable(
            map(lambda sample_tuple: self.boost_color_strat_sample(*sample_tuple), self._strategy_pool)
        ))

        # give the samples to the respective memories
        self._prediction_memory.add_samples(boosted_pred_pool, self._prediction_y_function(prediction, made_points))
        self._strategy_memory.add_samples(boosted_strat_pool, self._strategy_y_function(prediction, made_points)-discount)

        # potentially train
        if train:
            tmp = datetime.datetime.now()
            self._prediction_model.train_on_batch(*self._prediction_memory.draw_batch(self._batch_size_pred))
            self.strategy_model.train_on_batch(*self._strategy_memory.draw_batch(self._batch_size_strat))
            t = datetime.datetime.now() - tmp
            RnnPlayer.total_time_spent_in_keras += t
            RnnPlayer.time_spent_training += t

    def boost_color_pred_sample(self, model_input: np.ndarray) -> List[np.ndarray]:
        assert len(model_input.shape) == 1
        input_matrix = np.tile(model_input, (6, 1))
        suit_snippets = [model_input[i * 9: (i + 1) * 9] for i in range(4)]
        for row_index, swap in enumerate(swaps[1:], 1):
            matrix_row = input_matrix[row_index]
            for i, snippet_index in enumerate(swap[1: -1], 1):
                matrix_row[i * 9: (i + 1) * 9] = suit_snippets[snippet_index]
        return list(input_matrix)

    def boost_color_strat_sample(self, rnn_input: np.ndarray, aux_input: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ rnn_input is 2D and aux_input 1D, so this is the x of one training sample"""
        # rnn part
        rnn_cube = np.tile(rnn_input, (6, 1, 1))
        for cube_index, swap in enumerate(swaps[1:], 1):
            for blie in rnn_cube[cube_index]:
                for i in range(1, 8, 2):
                    blie[i] = swap[int(blie[i])]

        # aux part
        aux_matrix = np.tile(aux_input, (6, 1))
        suit_snippets = [aux_input[i * 9: (i + 1) * 9] for i in range(4)]
        for row_index, swap in enumerate(swaps[1:], 1):
            matrix_row = aux_matrix[row_index]
            for i, snippet_index in enumerate(swap[1: -1], 1):
                matrix_row[i * 9: (i + 1) * 9] = suit_snippets[snippet_index]
            matrix_row[36:44] = rnn_cube[row_index][-1][:8]
            matrix_row[-1] = swap[int(matrix_row[-1])]
        return [(rnn_cube[i], aux_matrix[i]) for i in range(6)]

    def _get_relative_rnn_input(self, state: GameState) -> np.ndarray:
        history_absolute = state.blies_history[:state.current_blie_index + 1]
        history_rolled = history_absolute[:, self._table_roll_indices]
        history_rolled[:, -1] = (history_rolled[:, -1] - self._table_position) % 4
        return history_rolled


class StreunRnnPlayer(DifferenzlerPlayer):

    # note: this player is purely to be able to compare earlier work with current work. So it will only ever
    # be run in agent_arena and not be used to train anything

    _prediction_model: keras.Model
    strategy_model: keras.Model

    _hand_vector: np.ndarray
    _table_position: int
    _table_roll_indices: np.ndarray

    def __init__(self, prediction_model: keras.Model, strategy_model: keras.Model):
        self._prediction_model = prediction_model
        self.strategy_model = strategy_model

    def start_round(self, hand_vector: np.ndarray, table_position: int):
        self._hand_vector = hand_vector
        self._table_position = table_position
        self._table_roll_indices = np.concatenate([np.roll(np.arange(8), -table_position * 2), [8]])

    def make_prediction(self) -> int:
        model_input = np.concatenate([
            self._hand_vector,
            np.array([self._table_position])
        ]).reshape((1, -1))
        return int(self._prediction_model.predict(model_input).reshape(-1)[0] + 0.5)

    def play_card(self, state: GameState, suit: int) -> np.ndarray:
        rnn_state_tensor, aux_state_action_tensor = self.form_nn_input_tensors(state, suit)
        if len(aux_state_action_tensor) == 0:
            q_values = []
        else:
            q_values = self.strategy_model.predict([rnn_state_tensor, aux_state_action_tensor])
        return self.get_action(q_values)

    _current_possible_actions: np.ndarray
    _current_rnn_state_tensors: np.ndarray
    _current_aux_state_action_tensors: np.ndarray

    def form_nn_input_tensors(self, state: GameState, suit: int) -> Tuple[np.ndarray, np.ndarray]:
        possible_actions = get_possible_actions(self._hand_vector, suit)
        nbr_of_actions = len(possible_actions)
        if np.sum(self._hand_vector) == 1:
            self._current_possible_actions = possible_actions
            return np.array([]), np.array([])
        rnn_state_tensor = np.tile(
            self._get_relative_rnn_input(state),
            (nbr_of_actions, 1, 1)
        )
        table = state.blies_history[state.current_blie_index][:8]
        current_difference = [state.predictions[self._table_position] - state.points_made[self._table_position]]
        abs_position_vector = np.zeros(4)
        abs_position_vector[self._table_position] = 1
        aux_state_tensor = np.tile(
            np.concatenate([abs_position_vector, self._hand_vector, table, state.gone_cards, current_difference]),
            (nbr_of_actions, 1)
        )
        aux_state_action_tensor = np.concatenate([aux_state_tensor, possible_actions], axis=1)
        self._current_possible_actions = possible_actions
        self._current_rnn_state_tensors = rnn_state_tensor
        self._current_aux_state_action_tensors = aux_state_action_tensor
        return rnn_state_tensor, aux_state_action_tensor

    def get_action(self, q_values: np.ndarray) -> np.ndarray:
        if len(q_values) == 0:
            action = self._current_possible_actions[0]
        else:
            index = np.argmax(q_values)
            action = self._current_possible_actions[index]
        self._hand_vector[action[0] + 9 * action[1]] = 0
        return action

    def finish_round(self, prediction: int, made_points: int, train: bool, discount: Union[int, float] = 0.0) -> Any:
        pass

    def _get_relative_rnn_input(self, state: GameState) -> np.ndarray:
        history_absolute = state.blies_history[:state.current_blie_index + 1]
        history_rolled = history_absolute[:, self._table_roll_indices]
        history_rolled[:, -1] = (history_rolled[:, -1] - self._table_position) % 4
        return history_rolled


class HandCraftEverywhereRnnPlayer(RnnPlayer):

    _small_roll_array: np.ndarray  # has 4 entries

    def start_round(self, hand_vector: np.ndarray, table_position: int):
        super().start_round(hand_vector, table_position)
        self._small_roll_array = np.roll(np.arange(4), -table_position)

    def form_nn_input_tensors(self, state: GameState, suit: int) -> Tuple[np.ndarray, np.ndarray]:
        possible_actions = get_possible_actions(self._hand_vector, suit)
        nbr_of_actions = len(possible_actions)
        if np.sum(self._hand_vector) == 1:
            self._current_possible_actions = possible_actions
            return np.array([]), np.array([])
        rnn_state_tensor = np.tile(
            self._get_relative_rnn_input(state),
            (nbr_of_actions, 1, 1)
        )
        relative_table = state.blies_history[state.current_blie_index][self._table_roll_indices[:8]]
        current_difference = [state.predictions[self._table_position] - state.points_made[self._table_position]]
        gone_cards = state.gone_cards
        bocks = which_are_bock(gone_cards, self._hand_vector)
        could_follow = state.get_could_follow_vector(self._small_roll_array)
        points_currently_on_table = np.array([state.get_points_on_table()])
        points_per_player = state.points_made[self._small_roll_array]
        aux_state_tensor = np.tile(
            np.concatenate([self._hand_vector, relative_table, current_difference, gone_cards, bocks, could_follow,
                            points_currently_on_table, points_per_player]),
            (nbr_of_actions, 1)
        )
        aux_state_action_tensor = np.concatenate([aux_state_tensor, possible_actions], axis=1)
        self._current_possible_actions = possible_actions
        self._current_rnn_state_tensors = rnn_state_tensor
        self._current_aux_state_action_tensors = aux_state_action_tensor
        return rnn_state_tensor, aux_state_action_tensor

    def boost_color_pred_sample(self, vector: np.ndarray) -> List[np.ndarray]:
        return list(boost_basic_prediction_vector(vector))

    def boost_color_strat_sample(self, rnn_part: np.ndarray, aux_part: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        return list(boost_hand_crafted_strategy_vector(rnn_part, aux_part))
