import numpy as np


def translate_vector_to_two_number_representation(hand: np.ndarray) -> np.ndarray:
    card_indices = np.nonzero(hand)[0]
    wanted_representation = [[x % 9, x // 9] for x in card_indices]
    return np.array(wanted_representation)


def color_trump(hand: np.ndarray, first_played_suit: int) -> np.ndarray:
    """
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


def direction_trump(hand: np.ndarray, first_played_suit: int) -> np.ndarray:
    """
    :param hand: 36 entry vector
    :param first_played_suit: in [-1, 4]
    :return: all possible actions in two number representation as np.ndarray (first axis are the different options)
    """
    assert first_played_suit in range(-1, 4), "the value is " + str(first_played_suit)
    if first_played_suit != -1 and np.any(hand[first_played_suit * 9: (first_played_suit + 1) * 9]):
        return translate_vector_to_two_number_representation(hand[first_played_suit * 9: (first_played_suit + 1) * 9])
    else:
        return translate_vector_to_two_number_representation(hand)


def winning_card_index_classic_trump(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: only here for a more consistent interface with the other 'winning_card' methods
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


def winning_card_index_obe_abe(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: only here for a more consistent interface with the other 'winning_card' methods
    :return: index of the winning hand
    """
    highest_index = first_played_index
    played_suit = table[first_played_index][1]
    for i in range(1, 4):
        j = (i + first_played_index) % 4
        current_rank, current_suit = table[j]
        if current_suit == played_suit and current_rank > table[highest_index][0]:
            highest_index = j
    return highest_index


def winning_card_index_unne_ufe(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: only here for a more consistent interface with the other 'winning_card' methods
    :return: index of the winning hand
    """
    highest_index = first_played_index
    played_suit = table[first_played_index][1]
    for i in range(1, 4):
        j = (i + first_played_index) % 4
        current_rank, current_suit = table[j]
        if current_suit == played_suit and current_rank < table[highest_index][0]:
            highest_index = j
    return highest_index


def winning_card_index_slalom_obe(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: index of this blie
    :return: index of the winning hand
    """
    if blie_index % 2 == 0:
        return winning_card_index_obe_abe(table, first_played_index, blie_index)
    else:
        return winning_card_index_unne_ufe(table, first_played_index, blie_index)


def winning_card_index_slalom_unne(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: index of this blie
    :return: index of the winning hand
    """
    if blie_index % 2 == 1:
        return winning_card_index_obe_abe(table, first_played_index, blie_index)
    else:
        return winning_card_index_unne_ufe(table, first_played_index, blie_index)


def winning_card_index_resien_slalom_obe_abe(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: index of this blie
    :return: index of the winning hand
    """
    if blie_index <= 5:
        return winning_card_index_obe_abe(table, first_played_index, blie_index)
    else:
        return winning_card_index_unne_ufe(table, first_played_index, blie_index)


def winning_card_index_resien_slalom_unne_ufe(table: np.ndarray, first_played_index: int, blie_index: int) -> int:
    """
    looks who won the hand
    :param table: list of cards in the two-number-representation
    :param first_played_index: index of the player that started this hand
    :param blie_index: index of this blie
    :return: index of the winning hand
    """
    if blie_index > 5:
        return winning_card_index_obe_abe(table, first_played_index, blie_index)
    else:
        return winning_card_index_unne_ufe(table, first_played_index, blie_index)
