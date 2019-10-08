import numpy as np
import os
import sys
from typing import List, Union, Tuple

import keras

from PlayerInterlayer import RnnPlayerInterlayer
from Memory import ReplayMemory, RnnReplayMemory
from Network import PredictionNetwork, RnnStrategyNetwork
from Player import RnnPlayer
from Sitting import Sitting
from main_helper_methods import normal_pred_y_func, normal_strat_y_func


def check_model_names():
    if not len(sys.argv) in [3, 5, 9]:
        print("the length of the input isn't admissible (is {})".format(len(sys.argv)))
        exit()

    for name in sys.argv[1:]:
        if not os.path.isfile(name):
            print("'{}' does not exist".format(name))


def get_model_tuples() -> List[Tuple[keras.Model, keras.Model]]:
    preliminary_list = [
        (keras.models.load_model(sys.argv[i]), keras.models.load_model(sys.argv[i+1]))
        for i in range(1, len(sys.argv), 2)
    ]
    print(preliminary_list[0][1].summary())
    return 4 // len(preliminary_list) * preliminary_list


def get_player(pred_model: keras.Model, strat_model: keras.Model) -> RnnPlayer:
    small_replay_memory = ReplayMemory(1)
    small_rnn_replay_memory = RnnReplayMemory(1)

    prediction_network = PredictionNetwork(pred_model, small_replay_memory, 1, False)
    strategy_network = RnnStrategyNetwork(strat_model, small_rnn_replay_memory, 1, False)

    return RnnPlayer(prediction_network, strategy_network, 0.0, 0.0)


def print_for_each_player(leading_text: str, data: Union[List, np.ndarray]):
    print('')
    print(leading_text)
    for i in range(len(data)):
        print("Player {} \t {}".format(i+1, data[i]))


def main():
    check_model_names()
    models = get_model_tuples()
    players = [RnnPlayerInterlayer(get_player(*model), normal_pred_y_func, normal_strat_y_func) for model in models]

    print("players have been loaded")

    sitting = Sitting(False)
    sitting.set_players(players)

    nbr_rounds = 10
    points_limit = 1_000
    total_diffs = np.zeros(4, dtype=np.float64)
    total_hands = 0
    wins = np.zeros(4)
    for round_index in range(nbr_rounds):
        round_diffs = np.zeros(4)
        while not np.any(round_diffs >= points_limit):
            points_made, predictions = sitting.play_cards()
            diffs = np.abs(points_made - predictions)
            round_diffs += diffs
            total_hands += 1
            for p in players:
                p.end_round(0, 0)
        total_diffs += round_diffs
        winner_index = np.argmin(round_diffs)
        wins[winner_index] += 1

    print("After {} rounds playing until a player has {} points.".format(nbr_rounds, points_limit))

    print_for_each_player("Wins:", wins)
    print_for_each_player("Avg diffs:", total_diffs / total_hands)


if __name__ == '__main__':
    main()
