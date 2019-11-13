import numpy as np

trump_points = {0: 20, 1: 14, 2: 11, 3: 4, 4: 3, 5: 10, 6: 0, 7: 0, 8: 0}
color_points = {0: 11, 1: 4, 2: 3, 3: 2, 4: 10, 5: 0, 6: 0, 7: 0, 8: 0}


def get_points_from_table(table: np.ndarray, last_round: bool) -> int:
    total = 5 if last_round else 0
    for card in table:
        total += color_points[card[0]] if card[1] > 0 else trump_points[card[0]]
    return total


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


def which_are_bock(gone_cards: np.ndarray, hand_cards: np.ndarray) -> np.ndarray:
    hand_cards_block = hand_cards.reshape((-1, 9))
    output = np.zeros(36)
    for suit in range(4):
        if not np.any(hand_cards_block[suit]):
            continue
        rank = int(np.argmax(hand_cards_block[suit]))
        if np.all(gone_cards[9*suit: 9*suit + rank]):
            for i in range(rank, 9):
                if hand_cards_block[suit, i] != 0:
                    output[suit*9+i] = 1
                else:
                    break
    return output
