import sys
from typing import List, Tuple

import keras
import numpy as np

from abstract_classes.player import DifferenzlerPlayer
from main_helper_methods import normal_pred_y_func, normal_strat_y_func, prediction_resnet, normal_strategy_network, \
    small_strategy_network, tiny_strategy_network, hand_crafted_features_rnn_network, \
    hand_crafted_features_rnn_network_wider, small_bidirectional_strategy_network, hand_crafted_features_hinton, \
    hand_crafted_features_double_hinton
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer, StreunRnnPlayer, HandCraftEverywhereRnnPlayer
from sitting import DifferenzlerSitting

number_of_parallel_table_configurations = 166
number_of_rounds = 10_000 // 6 // number_of_parallel_table_configurations


def get_net(name: str) -> keras.Model:
    if 'strat' in name:
        if "strat_hand_craft" in name and "wider" in name:
            return hand_crafted_features_rnn_network_wider()
        elif "strat_hand_craft" in name:
            return hand_crafted_features_rnn_network()
        elif 'hinton' in name:
            print("Be aware of the dropout rate")
            if 'double' in name:
                return hand_crafted_features_double_hinton(dropout=0.5)
            else:
                return hand_crafted_features_hinton(0.5)
        elif 'normal' in name:
            return normal_strategy_network()
        elif 'small' in name and 'bidirectional' in name:
            return small_bidirectional_strategy_network()
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
    return [(nets[i], nets[i + 1]) for i in range(0, len(nets), 2)]


def load_models_with_streun(net_names: List[str]) -> List[Tuple[keras.Model, keras.Model]]:
    nets = [get_net(net_name) for net_name in net_names]
    for i in range(2):
        nets[i].load_weights(net_names[i])
    nets.append(keras.models.load_model("./normal_prediction_2000000.h5"))
    nets.append(keras.models.load_model("./normal_strategy_2000000.h5"))
    return [(nets[i], nets[i + 1]) for i in range(0, len(nets), 2)]


def get_models_configs(models):
    t_model, s_model = models[0], models[1]
    all_configs = [
        t_model, s_model, t_model, s_model,
        s_model, t_model, s_model, t_model,
        t_model, t_model, s_model, s_model,
        s_model, t_model, t_model, s_model,
        s_model, s_model, t_model, t_model,
        t_model, s_model, s_model, t_model
    ]
    t_model_indices_base = np.array([0, 2, 5, 7, 8, 9, 13, 14, 18, 19, 20, 23])
    s_model_indices_base = np.array([1, 3, 4, 6, 10, 11, 12, 15, 16, 17, 21, 22])
    t_model_indices = [t_model_indices_base + 24 * np.ones(12) * k for k in
                       range(number_of_parallel_table_configurations)]
    s_model_indices = [s_model_indices_base + 24 * np.ones(12) * k for k in
                       range(number_of_parallel_table_configurations)]
    t_model_indices = np.array(t_model_indices, dtype=int).reshape(-1)
    s_model_indices = np.array(s_model_indices, dtype=int).reshape(-1)
    all_configs *= number_of_parallel_table_configurations
    return all_configs, s_model_indices, t_model_indices


def main():
    sys_inputs = [sys.argv[1:]]
    for sys_input in sys_inputs:
        use_streun = '-streun' in sys_input
        if use_streun:
            sys_input.remove('-streun')
        for i in sys_input:
            print(i)
        if use_streun:
            print("streun_pred")
            print("streun_strat")
        models = load_models_with_streun(sys_input) if use_streun else load_all_models(sys_input)
        all_configs, s_model_indices, t_model_indices = get_models_configs(models)
        pred_memory = ReplayMemory(1)
        strat_memory = RnnReplayMemory(1)
        players: List[DifferenzlerPlayer] = []
        if use_streun:
            players = [
                HandCraftEverywhereRnnPlayer(
                    pred_model, strat_model, pred_memory, strat_memory, normal_pred_y_func, normal_strat_y_func, 0.0,
                    0.0, 1, 1
                ) if pred_model == all_configs[0][0] else
                StreunRnnPlayer(pred_model, strat_model)
                for pred_model, strat_model in all_configs
            ]
        else:
            players_constr = [HandCraftEverywhereRnnPlayer, HandCraftEverywhereRnnPlayer]
            players = [
                (players_constr[0] if pred_model == all_configs[0][0] else players_constr[1])(
                    pred_model, strat_model, pred_memory, strat_memory, normal_pred_y_func, normal_strat_y_func,
                    0.0, 0.0, 1, 1
                )
                for pred_model, strat_model in all_configs
            ]

        sitting = DifferenzlerSitting()
        sitting.set_players(players)
        total_diffs = np.zeros(24 * number_of_parallel_table_configurations)
        won_rounds = np.zeros(24 * number_of_parallel_table_configurations)
        for i in range(number_of_rounds):
            preds, mades = sitting.play_cards(shuffle=False)
            diffs = np.absolute(preds - mades).reshape(-1)
            total_diffs += diffs
            for table_i, table_diffs in enumerate(diffs.reshape((-1, 4))):
                indices_of_wins = np.where(table_diffs == table_diffs.min())[0]
                for win_indice in indices_of_wins:
                    won_rounds[table_i * 4 + win_indice] += 1 / len(indices_of_wins)
            print("{}% ({} / {})".format(int((i+1) / number_of_rounds * 1000) / 10, i+1, number_of_rounds), end='\r')
        # total diffs contains the diffs of each player. Each player only has played 'number_of_rounds' rounds
        avg_diffs = total_diffs / number_of_rounds
        diffs_per_agent_type = [
            np.sum(avg_diffs[t_model_indices]) / len(t_model_indices),
            np.sum(avg_diffs[s_model_indices]) / len(s_model_indices)
        ]
        wins_per_agent_type = [
            np.sum(won_rounds[t_model_indices]),
            np.sum(won_rounds[s_model_indices])
        ]
        wa = (wins_per_agent_type[0] / (number_of_rounds * 6 * number_of_parallel_table_configurations) * 100 - 50) * 2
        # print("Average difference:", avg_diffs)
        # print("Won rounds:", won_rounds)
        print("avg. diffs: {0:.2f} vs {1:.2f}".format(diffs_per_agent_type[0], diffs_per_agent_type[1]))
        print("WA = {0:.2f}".format(wa))
        print("total rounds:", number_of_rounds * 6 * number_of_parallel_table_configurations)
        print("\n\n\n")


if __name__ == '__main__':
    main()
