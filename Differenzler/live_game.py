import numpy as np
import sys

import keras

from Memory import ReplayMemory, RnnReplayMemory
from Network import PredictionNetwork, RnnStrategyNetwork
from Player import RnnPlayer
from PlayerInterlayer import RnnPlayerInterlayer
from helper_functions import get_winning_card_index, get_points_from_table

rank_strings = ['a', 'k', 'd', 'j', 'z', '9', '8', '7', '6']
suit_strings = ['s', 'h', 'e', 'c']


def trump_rank_to_int(query: str) -> int:
    assert query in rank_strings
    tmp = rank_strings[3:4] + rank_strings[5:6] + rank_strings[:3] + rank_strings[4:5] + rank_strings[6:]
    assert len(tmp) == 9
    output = tmp.index(query)
    assert 0 <= output < 9
    return output


def color_rank_to_int(query: str) -> int:
    assert query in rank_strings
    output = rank_strings.index(query)
    assert 0 <= output < 9
    return output


def suit_string_to_int(suit_str: str) -> int:
    assert suit_str in suit_strings
    return suit_strings.index(suit_str)


def card_string_is_okay(card_string: str) -> bool:
    if card_string is None or len(card_string) != 2:
        return False
    elif card_string[0] in rank_strings and card_string[1] in suit_strings:
        return True
    return False


def translate_card_string_to_tnr(card_string: str) -> np.ndarray:
    assert card_string_is_okay(card_string)
    output = np.zeros(2, dtype=int)
    if card_string[1] == suit_strings[0]:
        output[0] = trump_rank_to_int(card_string[0])
    else:
        output[0] = color_rank_to_int(card_string[0])
    output[1] = suit_string_to_int(card_string[1])
    return output


def card_still_in(card_string: str, gone_cards: np.ndarray) -> bool:
    tnr = translate_card_string_to_tnr(card_string)
    return gone_cards[int(tnr[0] + 9 * tnr[1])] == 0


def print_play_message(card_tnr: np.ndarray):
    output_str = "The Network plays: "
    if card_tnr[1] == 0:
        tmp = rank_strings[3:4] + rank_strings[5:6] + rank_strings[:3] + rank_strings[4:5] + rank_strings[6:]
        output_str += tmp[card_tnr[0]]
    else:
        output_str += rank_strings[card_tnr[0]]
    output_str += suit_strings[card_tnr[1]]
    print(output_str)


assert len(sys.argv) == 3

pred_memory = ReplayMemory(1)
strat_memory = RnnReplayMemory(1)

pred_network = PredictionNetwork(keras.models.load_model(sys.argv[1]), pred_memory, batch_size=1, can_train=False)
strat_network = RnnStrategyNetwork(keras.models.load_model(sys.argv[2]), strat_memory, batch_size=1, can_train=False)

player = RnnPlayer(pred_network, strat_network, 0, 0)

absolute_position = int(input("What is the index of the player? "))
assert 0 <= absolute_position < 4, "the given absolute position is " + str(absolute_position)
player_inter = RnnPlayerInterlayer(player, sum, sum)
player_inter.set_absolute_position(absolute_position)

# get the trump suit
trump_string = None
while trump_string not in suit_strings:
    trump_string = input("The trump suit: ").strip().lower()
suit_strings.remove(trump_string)
suit_strings = [trump_string] + suit_strings
assert len(np.unique(suit_strings)) == 4

# get the hand of the network
hand_card_strings = []
while len(np.unique(hand_card_strings)) != 9 or \
        not np.all([card_string_is_okay(hand_card) for hand_card in hand_card_strings]):
    hand_card_strings = list(map(lambda x: x.strip(), input("Hand cards: ").split(" ")))
hand_cards_tnr = list(map(lambda x: translate_card_string_to_tnr(x), hand_card_strings))
hand_vector = np.zeros(36)
for hand_card_tnr in hand_cards_tnr:
    hand_vector[int(hand_card_tnr[0] + 9 * hand_card_tnr[1])] = 1
player_inter.receive_hand(hand_vector)

# get prediction
prediction = player_inter.make_prediction(absolute_position)
print("The network predicts:", prediction, "points")

blie_starter_index = 0
blie_history = []
points_made = np.zeros(4)
gone_cards = np.zeros(36)


for blie_index in range(9):
    print("\nBlie", blie_index + 1)
    table = np.ones((4, 2), dtype=int) * -1
    table_suit = -1
    for i in range(4):
        player_at_turn_index = (blie_starter_index + i) % 4
        if player_at_turn_index == absolute_position:
            card_tnr = player_inter.play_card(table, blie_starter_index, gone_cards,
                                              prediction - points_made[absolute_position], blie_history)
            print_play_message(card_tnr)
        else:
            card_string: str = None
            while not (card_string_is_okay(card_string) and card_still_in(card_string, gone_cards)):
                card_string = input("Card player " + str(player_at_turn_index) + " played: ").strip().lower()
            card_tnr = translate_card_string_to_tnr(card_string)
        gone_cards[int(card_tnr[0] + 9 * card_tnr[1])] = 1
        if table_suit < 0:
            table_suit = card_tnr[1]
        table[player_at_turn_index] = card_tnr

    winning_index = get_winning_card_index(table, blie_starter_index)
    blie_history.append((table, blie_starter_index))
    points_on_table = get_points_from_table(table, blie_index == 8)
    points_made[winning_index] += points_on_table
    blie_starter_index = winning_index

print(points_made)
assert np.all(gone_cards)
assert np.sum(points_made) == 157
