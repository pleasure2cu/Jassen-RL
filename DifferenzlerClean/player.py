import datetime
from typing import List, Tuple, Callable, Union

import keras
import numpy as np

from abstract_classes.player import DifferenzlerPlayer
from memory import ReplayMemory, RnnReplayMemory
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


def boost_color_pred_sample(model_input: np.ndarray) -> List[np.ndarray]:
    assert len(model_input.shape) == 1
    input_matrix = np.tile(model_input, (6, 1))
    suit_snippets = [model_input[i * 9: (i + 1) * 9] for i in range(4)]
    for row_index, swap in enumerate(swaps[1:], 1):
        matrix_row = input_matrix[row_index]
        for i, snippet_index in enumerate(swap[1: -1], 1):
            matrix_row[i * 9: (i + 1) * 9] = suit_snippets[snippet_index]
    return list(input_matrix)


def boost_color_strat_sample(rnn_input: np.ndarray, aux_input: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ rnn_input is 2D and aux_input 1D, so this is the x of one training sample"""
    # rnn part
    rnn_cube = np.tile(rnn_input, (6, 1, 1))
    for cube_index, swap in enumerate(swaps[1:], 1):
        for blie in rnn_cube[cube_index]:
            for i in range(1, 8, 2):
                blie[i] = swap[int(blie[i])]

    # aux part
    aux_matrix = np.tile(aux_input, (6, 1))
    suit_snippets = [aux_input[i*9: (i+1)*9] for i in range(4)]
    for row_index, swap in enumerate(swaps[1:], 1):
        matrix_row = aux_matrix[row_index]
        for i, snippet_index in enumerate(swap[1: -1], 1):
            matrix_row[i*9: (i+1)*9] = suit_snippets[snippet_index]
        matrix_row[36:44] = rnn_cube[row_index][-1][:8]
        matrix_row[-1] = swap[int(matrix_row[-1])]
    return [(rnn_cube[i], aux_matrix[i]) for i in range(6)]


class RnnPlayer(DifferenzlerPlayer):

    # TODO: write down what the input looks like

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
    _strategy_model: keras.Model
    _prediction_memory: ReplayMemory
    _strategy_memory: RnnReplayMemory
    _prediction_exp: float
    _strategy_exp: float
    _prediction_pool: List[np.ndarray]
    _strategy_pool: List[Tuple[np.ndarray, np.ndarray]]
    _prediction_y_function: Callable[[int, int], Union[int, float]]  # (angesagt, gemacht) -> |R
    _strategy_y_function: Callable[[int, int], Union[int, float]]  # (angesagt, gemacht) -> |R

    _hand_vector: np.ndarray
    _table_position: int
    _table_roll_indices: np.ndarray
    _batch_size: int

    total_time_spent_in_keras = datetime.timedelta()
    time_spent_training = datetime.timedelta()

    def __init__(
            self,
            prediction_model: keras.Model, strategy_model: keras.Model,
            prediction_memory: ReplayMemory, strategy_memory: RnnReplayMemory,
            prediction_y_function: Callable[[int, int], Union[int, float]],
            strategy_y_function: Callable[[int, int], Union[int, float]],
            prediction_exp: float, strategy_exp: float,
            batch_size: int
    ):
        self._prediction_model = prediction_model
        self._strategy_model = strategy_model
        self._prediction_memory = prediction_memory
        self._strategy_memory = strategy_memory
        self._prediction_exp = prediction_exp
        self._strategy_exp = strategy_exp
        self._prediction_y_function = prediction_y_function
        self._strategy_y_function = strategy_y_function
        self._batch_size = batch_size

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

    def play_card(self, state: GameState, suit: int) -> np.ndarray:
        possible_actions = get_possible_actions(self._hand_vector, suit)
        nbr_of_actions = len(possible_actions)
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

        # get the index of the action we want to play
        if np.random.binomial(1, self._strategy_exp):
            index = np.random.randint(nbr_of_actions)
        else:
            tmp = datetime.datetime.now()
            index = np.argmin(self._strategy_model.predict([rnn_state_tensor, aux_state_action_tensor]))
            RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp

        self._strategy_pool.append((rnn_state_tensor[index], aux_state_action_tensor[index]))
        action = possible_actions[index]
        self._hand_vector[action[0] + 9 * action[1]] = 0
        return action

    def finish_round(self, prediction: int, made_points: int, train: bool) -> Tuple[float, float]:
        assert np.all(self._hand_vector == 0)
        boosted_pred_pool = []
        for sample in self._prediction_pool:
            boosted_pred_pool += boost_color_pred_sample(sample.reshape(-1))
        self._prediction_memory.add_samples(boosted_pred_pool, self._prediction_y_function(made_points))
        boosted_strat_pool = []
        for rnn_input, aux_input in self._strategy_pool:
            boosted_strat_pool += boost_color_strat_sample(rnn_input, aux_input)
        self._strategy_memory.add_samples(boosted_strat_pool, self._strategy_y_function(prediction, made_points))
        if train:
            tmp = datetime.datetime.now()
            pred_loss = self._prediction_model.train_on_batch(*self._prediction_memory.draw_batch(self._batch_size))
            strat_loss = self._strategy_model.train_on_batch(*self._strategy_memory.draw_batch(self._batch_size))
            RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
            RnnPlayer.time_spent_training += datetime.datetime.now() - tmp
            return pred_loss, strat_loss
        return 0.0, 0.0

    # def finish_round(self, prediction: int, made_points: int, train: bool) -> Tuple[float, float]:
    #     assert np.all(self._hand_vector == 0)
    #     self._prediction_memory.add_samples(self._prediction_pool, self._prediction_y_function(made_points))
    #     self._strategy_memory.add_samples(self._strategy_pool, self._strategy_y_function(prediction, made_points))
    #     if train:
    #         pred_loss = self._prediction_model.train_on_batch(*self._prediction_memory.draw_batch(self._batch_size))
    #         strat_loss = self._strategy_model.train_on_batch(*self._strategy_memory.draw_batch(self._batch_size))
    #         return pred_loss, strat_loss
    #     return 0.0, 0.0

    def _get_relative_rnn_input(self, state: GameState) -> np.ndarray:
        history_absolute = state.blies_history[:state.current_blie_index + 1]
        history_rolled = history_absolute[:, self._table_roll_indices]
        history_rolled[:, -1] = (history_rolled[:, -1] - self._table_position) % 4
        return history_rolled
