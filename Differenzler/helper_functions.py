from typing import Union, Tuple, List

import keras
import numpy as np

from Sample import RnnSample

TNRepresentation = np.array
Sample = Tuple[np.array, Union[int, float]]

trump_points = {0: 20, 1: 14, 2: 11, 3: 4, 4: 3, 5: 10, 6: 0, 7: 0, 8: 0}
color_points = {0: 11, 1: 4, 2: 3, 3: 2, 4: 10, 5: 0, 6: 0, 7: 0, 8: 0}


def translate_vector_to_two_number_representation(hand: np.array) -> np.ndarray:
    card_indices = np.nonzero(hand)[0]
    wanted_representation = [[x % 9, x // 9] for x in card_indices]
    return np.array(wanted_representation)


def get_all_possible_actions(hand: np.array, first_played_suit: int) -> np.ndarray:
    """
    This function gets all the cards that are allowed to be played according to the rules of the game
    and returns them in the two-numbers representation
    :param hand: 36 entry vector
    :param first_played_suit: in [-1, 4]
    :return: all possible actions in two number representation as np.ndarray (first axis are the different options)
    """
    assert first_played_suit in range(-1, 4), "the value is " + str(first_played_suit)
    if first_played_suit == 0:
        if np.any(hand[1: 9]):
            playable_cards = hand[:9]
        else:
            playable_cards = hand
    elif np.any(hand[first_played_suit * 9: (first_played_suit + 1) * 9]):
        playable_cards = np.zeros(36)
        playable_cards[:9] = hand[:9]
        playable_cards[first_played_suit * 9: (first_played_suit + 1) * 9] = \
            hand[first_played_suit * 9: (first_played_suit + 1) * 9]
    else:
        playable_cards = hand
    return translate_vector_to_two_number_representation(playable_cards)


def get_winning_card_index(table: np.ndarray, first_played_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :return: index of the winning hand
    """
    highest_index = first_played_index
    for i in range(1, 4):
        j = (i + first_played_index) % 4
        if table[highest_index][1] != 0 and table[j][1] == 0:  # trump after non-trump
            highest_index = j
        elif table[highest_index][1] == table[j][1] and table[j][0] < table[highest_index][0]:
            highest_index = j
    return highest_index


def get_points_from_table(table: np.ndarray, last_round: bool) -> int:
    total = 5 if last_round else 0
    for card in table:
        total += color_points[card[0]] if card[1] > 0 else trump_points[card[0]]
    return total


def turn_rnn_samples_into_batch(samples: List[RnnSample]) -> Tuple[np.ndarray, np.ndarray, np.array]:
    assert len(samples) > 0

    n = len(samples)
    rnn_input_batch = np.zeros((n,) + samples[0].rnn_input.shape)
    aux_input_batch = np.zeros((n,) + samples[0].aux_input.shape)
    y_batch = np.zeros(n)
    for i in range(n):
        sample = samples[i]
        rnn_input_batch[i] = sample.rnn_input
        aux_input_batch[i] = sample.aux_input
        y_batch[i] = sample.y

    return rnn_input_batch, aux_input_batch, y_batch


def resnet_block(input_tensor, layer_size: int, use_batch_norm: bool):
    """
    implements one resnet block. Meaning:
                       _______________________________________
                      |                                       |
        Input_tensor ---> Dense -> BN -> ReLU -> Dense -> BN ---> ReLU

    the resulting tensor of the last ReLU will be the return
    :param input_tensor: as name says
    :param layer_size: size of the output of the input_tensor
    :param use_batch_norm: bool-flag
    :return: tensor from the last ReLU
    """
    block = keras.layers.Dense(layer_size)(input_tensor)
    if use_batch_norm:
        block = keras.layers.BatchNormalization()(block)
    block = keras.layers.Activation('relu')(block)
    block = keras.layers.Dense(layer_size)(block)
    if use_batch_norm:
        block = keras.layers.BatchNormalization()(block)
    block = keras.layers.add([block, input_tensor])
    return keras.layers.Activation('relu')(block)


def two_nbr_rep_swap_suit(card: TNRepresentation, swap_suit1: int, swap_suit2: int) -> TNRepresentation:
    """
    expects a card in two-numbers representation. If the suit is equal to 'swap_suit1' or 'swap_suit2' the suits
    will be changed
    :param card: card to be examined
    :param swap_suit1: one of the suits that should be swapped
    :param swap_suit2: one of the suits that should be swapped
    :return: a card in two-number representation but with swapped suits (given that 'card' is of any of the suits)
    """
    assert card.shape == (2,)
    assert 0 <= swap_suit1 < 4
    assert 0 <= swap_suit2 < 4
    if swap_suit1 == 0 or swap_suit2 == 0:
        print('WARNING: you are swapping the trump suit')
    output = np.copy(card)
    if card[1] == swap_suit1:
        output[1] = swap_suit2
    elif card[1] == swap_suit2:
        output[1] = swap_suit1
    return output


def two_nbr_rep_table_booster(table: np.array, swap_suit1: int, swap_suit2: int) -> np.array:
    """
    expects a table in two-number-representation and swaps the two provided suits of all cards
    :param table: vector with 8 entries
    :param swap_suit1: one of the suits that should be swapped
    :param swap_suit2: one of the suits that should be swapped
    :return: a table in two-numbers representation but with swapped suits
    """
    assert table.shape == (8,)
    assert np.all(-1 <= table[[0, 2, 4, 6]]) and np.all(table[[0, 2, 4, 6]] < 9), "the ranks of the cards are out " \
                                                                                  "of range.\n" + str(table)
    assert np.all(-1 <= table[[1, 3, 5, 7]]) and np.all(table[[1, 3, 5, 7]] < 4), "the suits of the cards are out " \
                                                                                  "of range.\n" + str(table)
    new_table = np.zeros(8)
    for i in range(0, 8, 2):
        new_table[i: i + 2] = two_nbr_rep_swap_suit(table[i: i + 2], swap_suit1, swap_suit2)
    return new_table


def vector_rep_booster(vector: np.array, swap_suit1: int, swap_suit2: int) -> np.array:
    """
    expects a 36 entry vector and swaps the two given suits
    :param vector: 36 entry vector
    :param swap_suit1: one of the suits that should be swapped
    :param swap_suit2: one of the suits that should be swapped
    :return: a copy of 'vector' with the suits swapped
    """
    assert vector.shape == (36,)
    assert np.all(0 <= vector) and np.all(vector < 2), "the doesn't carry the expected representation:\n" + str(vector)
    assert 0 <= swap_suit1 < 4
    assert 0 <= swap_suit2 < 4
    swap_indices = np.arange(36)
    swap_indices[swap_suit1 * 9: (swap_suit1 + 1) * 9] = np.arange(swap_suit2 * 9, (swap_suit2 + 1) * 9)
    swap_indices[swap_suit2 * 9: (swap_suit2 + 1) * 9] = np.arange(swap_suit1 * 9, (swap_suit1 + 1) * 9)
    new_vector = vector[swap_indices]
    return new_vector


def state_action_83_booster(state_action_vector: np.array) -> List[np.array]:
    """
    this function expects a state-action vector and outputs all permutations that are equivalent from a strategy point
    of view. The expected scheme is as follows (sav := state_action_vector):
      sav[ 0: 8] - table cards in the two-numbers representation
      sav[ 8:44] - hand cards in the vector representation
      sav[44:80] - the gone cards in the vector representation
      sav[80:81] - will not be changed (how close player is to the prediction at the moment)
      sav[81:83] - the action in the two-numbers representation
    :param state_action_vector: the vector described above
    :return: a list of all vector versions that are the same from a strategy point of view
    """
    assert state_action_vector.shape == (83,)
    output = [np.copy(state_action_vector)]

    def tmp_f(vector, s1, s2):
        new_vector = np.zeros(83)
        new_vector[:8] = two_nbr_rep_table_booster(vector[:8], s1, s2)
        new_vector[8:44] = vector_rep_booster(vector[8:44], s1, s2)
        new_vector[44:80] = vector_rep_booster(vector[44:80], s1, s2)
        new_vector[80] = vector[80]
        new_vector[81:83] = two_nbr_rep_swap_suit(vector[81:83], s1, s2)
        output.append(new_vector)

    for i in range(1, 4):
        for j in range(i + 1, 4):
            tmp_f(state_action_vector, i, j)

    tmp_f(output[3], 1, 3)
    tmp_f(output[1], 1, 3)

    return output


def time_series_booster(series: np.ndarray) -> List[np.ndarray]:
    assert len(series.shape) == 2
    output = [series]

    def tmp_f(vector, i, j):
        tmp = np.zeros_like(vector)
        for k in range(len(vector)):
            tmp[k] = np.concatenate([two_nbr_rep_table_booster(vector[k][:8], i, j), vector[k][-1:]])
        output.append(tmp)

    for s1 in range(1, 4):
        for s2 in range(s1 + 1, 4):
            if s1 == s2:
                continue
            tmp_f(series, s1, s2)

    tmp_f(output[3], 1, 3)
    tmp_f(output[1], 1, 3)

    return output


def rnn_sample_booster(rnn_sample: RnnSample) -> List[RnnSample]:
    rnn_vecotrs = time_series_booster(rnn_sample.rnn_input)
    aux_vectors = np.concatenate([
        np.tile(rnn_sample.aux_input[:4], (6, 1)),
        state_action_83_booster(rnn_sample.aux_input[4:])
    ], axis=1)

    output = [RnnSample(rnn_vecotrs[i], aux_vectors[i], rnn_sample.y) for i in range(6)]
    return output


def prediction_state_37_booster(state_vector: np.array) -> List[np.array]:
    """
    takes the state vector used for prediction and outputs all versions that are the same from a strategy point of view
    the vector is assumed to have the following schema:
        sv[:36] - the hand cards in vector representation
        sv[36] - the position at the table
    :param state_vector: the 37 elements state vector
    :return: list of all state vectors that are the same for strategy considerations
    """
    output = []
    indices = np.arange(37, dtype=np.int)
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                if len(np.unique([i, j, k])) != 3:
                    continue
                indices[9: 18] = np.arange(i*9, (i+1)*9, dtype=np.int)
                indices[18: 27] = np.arange(j*9, (j+1)*9, dtype=np.int)
                indices[27: 36] = np.arange(k*9, (k+1)*9, dtype=np.int)
                addition = state_vector[indices]
                output.append(addition)
    return output
