from typing import Union, Tuple

import keras
import numpy as np


TNRepresentation = np.array
Sample = Tuple[np.array, Union[int, float]]


trump_points = {0: 20, 1: 14, 2: 11, 3: 4, 4: 3, 5: 10, 6: 0, 7: 0, 8: 0}
color_points = {0: 11, 1: 4, 2: 3, 3: 2, 4: 10, 5: 0, 6: 0, 7: 0, 8: 0}


def translate_vector_to_two_number_representation(hand: np.array) -> np.ndarray:
    card_indices = np.nonzero(hand)[0]
    wanted_representation = map(lambda x: [x % 9, x // 9], card_indices)
    return np.array(list(wanted_representation))


def get_all_possible_actions(hand: np.array, first_played_suit: int) -> np.ndarray:
    """
    This function gets all the cards that are allowed to be played according to the rules of the game
    and returns them in the two-numbers representation
    :param hand: 36 entry vector
    :param first_played_suit: in [-1, 4]
    :return: all possible actions in two number representation as np.ndarray (first axis are the different options)
    """
    if first_played_suit < 0 or not np.sum(hand[first_played_suit * 9: (first_played_suit + 1) * 9]):
        playable_cards = hand
    elif first_played_suit == 0:  # trump is played
        if np.sum(hand[first_played_suit: first_played_suit + 9]) == 1 and hand[0]:  # we only have the buur
            playable_cards = hand
        else:
            playable_cards = hand[:9]
    else:  # trump isn't played and we have the played suit
        playable_cards = np.zeros(36)
        playable_cards[:9] = hand[:9]
        playable_cards[first_played_suit * 9: (first_played_suit + 1) * 9] = \
            hand[first_played_suit * 9: (first_played_suit + 1) * 9]
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


def resnet_block(input_tensor, layer_size: int, use_batch_norm: bool):
    """
    implements one resnet block. Meaning:
        Input_tensor ---> Dense -> BN -> ReLU -> Dense -> BN ---> ReLU
                      |                                       |
                       ---------------------------------------
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
