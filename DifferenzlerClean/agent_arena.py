import sys
from typing import List, Tuple

import keras
import numpy as np

from abstract_classes.player import DifferenzlerPlayer
from main_helper_methods import normal_pred_y_func, normal_strat_y_func, prediction_resnet, normal_strategy_network, \
    small_strategy_network, tiny_strategy_network, hand_crafted_features_rnn_network, \
    hand_crafted_features_rnn_network_wider, small_bidirectional_strategy_network, hand_crafted_features_hinton, \
    hand_crafted_features_double_hinton, hand_crafted_features_quad_hinton
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer, StreunRnnPlayer, HandCraftEverywhereRnnPlayer
from sitting import DifferenzlerSitting

total_rounds = 20_000
rounds_per_partie = 20
starting_clock = 4  # there are four positions from which a hand can start
player_factor = rounds_per_partie // starting_clock  # we vectorise over the different positions of the starting player
parallel_factor = 5
epochs = total_rounds // rounds_per_partie // 2 // parallel_factor

# for readability
actual_diffs = 4

starting_clock_turn_back_indices = np.array([
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2]
])


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
            elif 'quad' in name:
                return hand_crafted_features_quad_hinton()
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
    cross = [
        t_model, s_model, t_model, s_model,
        s_model, t_model, s_model, t_model,
        t_model, s_model, t_model, s_model,
        s_model, t_model, s_model, t_model,
    ]
    side = [
        t_model, t_model, s_model, s_model,
        s_model, t_model, t_model, s_model,
        s_model, s_model, t_model, t_model,
        t_model, s_model, s_model, t_model
    ]
    return cross * player_factor * parallel_factor + side * player_factor * parallel_factor


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
        all_configs = get_models_configs(models)
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
            players_constr = [RnnPlayer, RnnPlayer]
            players = [
                (players_constr[0] if pred_model == all_configs[0][0] else players_constr[1])(
                    pred_model, strat_model, pred_memory, strat_memory, normal_pred_y_func, normal_strat_y_func,
                    0.0, 0.0, 1, 1
                )
                for pred_model, strat_model in all_configs
            ]

        sitting = DifferenzlerSitting()
        sitting.set_players(players)
        wins = [0, 0]
        for i in range(epochs):
            preds, mades = sitting.play_cards(shuffle=False)
            diffs = np.absolute(preds - mades).reshape((-1, actual_diffs))

            for k in range(len(diffs)):
                diffs[k] = diffs[k][starting_clock_turn_back_indices[k % 4]]

            diffs = diffs.reshape((2, parallel_factor, player_factor * starting_clock, actual_diffs))
            partie_diffs = np.sum(diffs, axis=2).reshape((-1, 4))

            for partie in partie_diffs:
                winner_indices = np.argwhere(partie == np.min(partie))[0]
                for winner_i in winner_indices:
                    wins[int(winner_i) % 2] += 1. / len(winner_indices)

            print("{}% ({} / {})".format(int((i + 1) / epochs * 1000) / 10, i + 1, epochs), end='\r')

        print("the total match ended with:\n{} - {}\tp({}, {}) = {}".format(
            wins[0], wins[1], rounds_per_partie,
            str(total_rounds / 1000) + 'K', wins[0] / total_rounds * rounds_per_partie
        ))
        print('over {} total rounds'.format(total_rounds))


if __name__ == '__main__':
    main()
