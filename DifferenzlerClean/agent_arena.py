import sys
from typing import List, Tuple

import keras
import numpy as np

from main_helper_methods import normal_pred_y_func, normal_strat_y_func, prediction_resnet, normal_strategy_network, \
    small_strategy_network, tiny_strategy_network
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer
from sitting import DifferenzlerSitting


number_of_rounds = 10_000


def get_net(name: str) -> keras.Model:
    if 'strat' in name:
        if 'normal' in name:
            return normal_strategy_network()
        elif "small" in name:
            return small_strategy_network()
        elif "tiny" in name:
            return tiny_strategy_network()
    else:
        return prediction_resnet()


def load_all_models(net_names: List[str]) -> List[Tuple[keras.Model, keras.Model]]:
    nets = [get_net(net_name) for net_name in net_names]
    for i in range(4):
        nets[i].load_weights(net_names[i])
    nets *= 8 // len(net_names)
    return [(nets[i], nets[i + 1]) for i in range(0, len(nets), 2)]


sys_inputs = [
    [
        "second round nets/pred_normal_player_45000.h5", "second round nets/strat_normal_player_45000.h5",
        "second round nets/pred_small_player_45000.h5", "second round nets/strat_small_player_45000.h5",
    ],
    [
        "second round nets/pred_normal_player_45000.h5", "second round nets/strat_normal_player_45000.h5",
        "second round nets/pred_tiny_player_45000.h5", "second round nets/strat_tiny_player_45000.h5",
    ],
    [
        "second round nets/pred_small_player_45000.h5", "second round nets/strat_small_player_45000.h5",
        "second round nets/pred_tiny_player_45000.h5", "second round nets/strat_tiny_player_45000.h5",
    ],
    [
        "second round nets/pred_small_player_not_boosted_45000.h5", "second round nets/strat_small_player_not_boosted_45000.h5",
        "second round nets/pred_small_player_45000.h5", "second round nets/strat_small_player_45000.h5",
    ],
    [
        "pred_normal_player_0_discount_150000.h5", "strat_normal_player_0_discount_150000.h5",
        "pred_normal_player_16_discount_150000.h5", "strat_normal_player_16_discount_150000.h5"
    ],
    [
        "pred_normal_player_0_discount_150000.h5", "strat_normal_player_0_discount_150000.h5",
        "pred_normal_player_32_discount_150000.h5", "strat_normal_player_32_discount_150000.h5"
    ],
    [
        "pred_normal_player_16_discount_150000.h5", "strat_normal_player_16_discount_150000.h5",
        "pred_normal_player_32_discount_150000.h5", "strat_normal_player_32_discount_150000.h5",
    ],
]


def main():
    for sys_input in sys_inputs:
        for i in sys_input:
            print(i)
        models = load_all_models(sys_input)
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
            diffs = np.absolute(preds - mades).reshape(-1)
            total_diffs += diffs
            indices_of_wins = np.where(diffs == diffs.min())[0]
            for win_i in indices_of_wins:
                won_rounds[win_i] += 1 / len(indices_of_wins)
            print("{}% ({} / {})".format(int((i+1) / number_of_rounds * 1000) / 10, i+1, number_of_rounds), end='\r')
        avg_diffs = total_diffs / number_of_rounds
        diffs_per_agent_type = [(avg_diffs[0] + avg_diffs[2])/2, (avg_diffs[1] + avg_diffs[3])/2]
        wins_per_agent_type = [won_rounds[0] + won_rounds[2], won_rounds[1] + won_rounds[3]]
        wa = (max(wins_per_agent_type) / number_of_rounds * 100 - 50) * 2
        print("Average difference:", avg_diffs)
        print("Won rounds:", won_rounds)
        print("avg. diffs: {0:.2f} vs {1:.2f}".format(diffs_per_agent_type[0], diffs_per_agent_type[1]))
        print("WA = {0:.2f}".format(wa))
        print("\n\n\n")


if __name__ == '__main__':
    main()
