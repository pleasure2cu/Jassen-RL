import sys
from typing import Tuple

import keras
import numpy as np

from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer
from sitting import get_winning_card_index, get_points_from_table
from state import GameState

rank_strings = ['a', 'k', 'd', 'j', 'z', '9', '8', '7', '6']
suit_strings = ['s', 'h', 'e', 'c']


def trump_rank_to_int(query: str) -> int:
    tmp = rank_strings[3:4] + rank_strings[5:6] + rank_strings[:3] + rank_strings[4:5] + rank_strings[6:]
    return tmp.index(query)


def color_rank_to_int(query: str) -> int:
    return rank_strings.index(query)


def suit_string_to_int(suit_str: str) -> int:
    return suit_strings.index(suit_str)


def translate_card_string_to_tnr(card_string: str) -> np.ndarray:
    output = np.zeros(2, dtype=int)
    if card_string[1] == suit_strings[0]:
        output[0] = trump_rank_to_int(card_string[0])
    else:
        output[0] = color_rank_to_int(card_string[0])
    output[1] = suit_string_to_int(card_string[1])
    return output


def load_models() -> Tuple[keras.Model, keras.Model]:
    return (
        keras.models.load_model(sys.argv[1]),
        keras.models.load_model(sys.argv[2])
    )


def get_hand_vector() -> np.ndarray:
    cards = []
    while len(cards) != 9:
        cards = input("Hand cards: ").split(" ")
    tnr_hand = map(translate_card_string_to_tnr, cards)
    indices = [card[0] + card[1] * 9 for card in tnr_hand]
    output = np.zeros(36)
    output[indices] = 1
    return output


def get_trump():
    global suit_strings
    trump_string = "ç"
    while trump_string not in suit_strings:
        trump_string = input("Trump: ")
    suit_strings.remove(trump_string)
    suit_strings = [trump_string] + suit_strings


def tnr_to_string(played_card: np.ndarray) -> str:
    output_str = ""
    if played_card[1] == 0:
        tmp = rank_strings[3:4] + rank_strings[5:6] + rank_strings[:3] + rank_strings[4:5] + rank_strings[6:]
        output_str += tmp[played_card[0]]
    else:
        output_str += rank_strings[played_card[0]]
    output_str += suit_strings[played_card[1]]
    return output_str


def get_played_card(player_index: int) -> np.ndarray:
    card_string = ""
    while len(card_string) != 2:
        card_string = input("Player {}: ".format(player_index + 1))
    return translate_card_string_to_tnr(card_string)


pred_model, strat_model = load_models()
player = RnnPlayer(pred_model, strat_model, ReplayMemory(1), RnnReplayMemory(1), sum, sum, 0.0, 0.0, 1)
state = GameState()

get_trump()
hand_card_vector = get_hand_vector()
player_index = int(input("Player position: "))
player.start_round(hand_card_vector, player_index)
player_prediction = player.make_prediction()
print("Player predicts:", player_prediction)
predictions = np.zeros(4)
predictions[player_index] = player_prediction
state.predictions = predictions


current_player = 0
for blie_index in range(9):
    print("\nBlie", blie_index + 1)
    current_suit = -1
    state.current_blie_index = blie_index
    state.set_starting_player_of_blie(current_player)
    for _ in range(4):
        if current_player == player_index:
            played_card = player.play_card(state, current_suit)
            print("Computer plays:", tnr_to_string(played_card))
        else:
            played_card = get_played_card(current_player)
        state.add_card(played_card, current_player)
        if current_suit < 0:
            current_suit = played_card[1]
        current_player += 1
        current_player %= 4
    table = np.reshape(state.blies_history[state.current_blie_index][:8], (4, 2))
    winning_index = get_winning_card_index(table, current_player)
    points_on_table = get_points_from_table(table, blie_index == 8)
    state.points_made[winning_index] += points_on_table
    current_player = winning_index

print(state.points_made)
