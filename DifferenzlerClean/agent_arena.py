import sys
from typing import List, Tuple

import keras
import numpy as np

from main_helper_methods import normal_pred_y_func, normal_strat_y_func, prediction_resnet, normal_strategy_network, \
    small_strategy_network, tiny_strategy_network
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer
from sitting import DifferenzlerSitting


number_of_rounds = 5_000


def load_all_models() -> List[Tuple[keras.Model, keras.Model]]:
    net_names = sys.argv[1:]
    if not len(net_names) in [2, 4, 8]:
        print("This number of nets is not admissible")
        exit()
    nets = [
        prediction_resnet(),
        small_strategy_network(),
        prediction_resnet(),
        small_strategy_network()
    ]
    for i in range(4):
        nets[i].load_weights(sys.argv[i+1])
    nets *= 8 // len(net_names)
    return [(nets[i], nets[i + 1]) for i in range(0, len(nets), 2)]


def main():
    models = load_all_models()
    pred_memory = ReplayMemory(1)
    strat_memory = RnnReplayMemory(1)
    players = [
        RnnPlayer(
            pred_model, strat_model, pred_memory, strat_memory, normal_pred_y_func, normal_strat_y_func, 0.0, 0.0, 1, 1
        )
        for pred_model, strat_model in models
    ]

    sitting = DifferenzlerSitting()
    sitting.set_players(players)
    total_diffs = np.zeros(4)
    won_rounds = np.zeros(4)
    for i in range(number_of_rounds):
        preds, mades = sitting.play_cards()
        diffs = np.absolute(preds - mades)
        total_diffs += diffs.reshape(-1)
        won_rounds[np.argmin(diffs)] += 1
        print("{}% ({} / {})".format(int(i+1 / number_of_rounds * 1000) / 10, i+1, number_of_rounds), end='\r')
    print("Average difference:", total_diffs / number_of_rounds)
    print("Won rounds:", won_rounds)


if __name__ == '__main__':
    main()
