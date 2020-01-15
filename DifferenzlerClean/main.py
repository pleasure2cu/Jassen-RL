import datetime
import os
from os import path
from typing import List, Tuple

import keras
import numpy as np

from abstract_classes.memory import Memory
from abstract_classes.player import DifferenzlerPlayer
from main_helper_methods import prediction_resnet, strategy_deep_lstm_resnet, normal_pred_y_func, normal_strat_y_func, \
    normal_strategy_network, small_strategy_network, tiny_strategy_network, \
    small_rnn_strategy_network, hand_crafted_features_rnn_network, \
    hand_crafted_features_rnn_network_wider, small_bidirectional_strategy_network, hand_crafted_features_hinton, \
    hand_crafted_features_double_hinton, hand_crafted_features_quad_hinton, aggressive_strat_y_func, \
    defensive_strat_y_func, very_aggressive_strat_y_func, very_defensive_strat_y_func
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer, HandCraftEverywhereRnnPlayer, StreunRnnPlayer
from sitting import DifferenzlerSitting

number_of_epochs = 5  # decides how many times the intermediate stats are written
epoch_size = 20_000  # decides over how many rounds an intermediate stats text goes
fit_window = 15  # after how many rounds the model is trained
sample_coverage = 1.0  # what percentage of samples do you want to be looked at (in the optimal case)
batch_size_strat = 192
sample_limit_strat = int(6 * 32 * fit_window * sample_coverage / batch_size_strat + 1) * batch_size_strat
batch_size_pred = int(batch_size_strat / 8 + 1)
sample_limit_pred = int(6 * 4 * fit_window * sample_coverage / batch_size_pred + 1) * batch_size_pred
print("Batch size for strat = {}".format(batch_size_strat))
print("Sample limit strategy = {}".format(sample_limit_strat))


def some_magic(discount: int) \
        -> Tuple[List[DifferenzlerPlayer], List[Tuple[keras.Model, Memory, keras.Model, Memory, int]], str]:
    dropout = 0.5

    pred_memory = ReplayMemory(2_000 * 6)
    strat_memory = RnnReplayMemory(16_000 * 6)

    pred_model: keras.Model = prediction_resnet()
    strat_model = hand_crafted_features_quad_hinton(dropout=0.5)
    strat_model.summary()

    # streun_pred_model = keras.models.load_model("./normal_prediction_2000000.h5")
    # streun_strat_model = keras.models.load_model("./normal_strategy_2000000.h5")
    #
    # players = [
    #     HandCraftEverywhereRnnPlayer(
    #         pred_model, strat_model, pred_memory, strat_memory,
    #         normal_pred_y_func, normal_strat_y_func, 0.07, 0.07, batch_size_pred, batch_size_strat
    #     ) if i % 30 in [0, 2, 5, 7, 8, 10, 13, 15, 14, 15, 19, 20, 24, 25, 26, 29] else
    #     StreunRnnPlayer(streun_pred_model, streun_strat_model)
    #     for i in range(4 * fit_window)
    # ]

    players = [
        HandCraftEverywhereRnnPlayer(
            pred_model, strat_model, pred_memory, strat_memory,
            normal_pred_y_func, normal_strat_y_func, 0.07, 0.07, batch_size_pred, batch_size_strat
        )
        for _ in range(4 * fit_window)
    ]

    return players, [(pred_model, pred_memory, strat_model, strat_memory, 1)], \
           "quad_hinton_net_new_{}_discount_{}_dropout_player".format(discount, int(dropout * 100))


def add_frozen_players(players: List[DifferenzlerPlayer]):
    path_to_frozen_nets = './ongoing_nets/forzen_nets/'
    frozen_nets = os.listdir(path_to_frozen_nets) if path.exists(path_to_frozen_nets) else []
    if len(frozen_nets) == 0:
        print("there are no frozen nets")
        return
    pred_mem = ReplayMemory(1)
    strat_mem = RnnReplayMemory(1)
    if any(map(lambda name: name.endswith('7995.h5'), frozen_nets)):  # we passed the first checkpoint
        net_names = [
            (
                path_to_frozen_nets + "pred_double_hinton_aggressive_discount_4_dropout_50_player_7995.h5",
                path_to_frozen_nets + "strat_double_hinton_aggressive_discount_4_dropout_50_player_7995.h5"
            ),
            (
                path_to_frozen_nets + "pred_double_hinton_defensive_discount_4_dropout_50_player_7995.h5",
                path_to_frozen_nets + "strat_double_hinton_defensive_discount_4_dropout_50_player_7995.h5"
            ),
            (
                path_to_frozen_nets + "pred_double_hinton_hyper_aggressive_discount_4_dropout_50_player_7995.h5",
                path_to_frozen_nets + "strat_double_hinton_hyper_aggressive_discount_4_dropout_50_player_7995.h5"
            ),
            (
                path_to_frozen_nets + "pred_double_hinton_hyper_defensive_discount_4_dropout_50_player_7995.h5",
                path_to_frozen_nets + "strat_double_hinton_hyper_defensive_discount_4_dropout_50_player_7995.h5"
            )
        ]
        for pred_name, strat_name in net_names:
            pred_model = prediction_resnet()
            pred_model.load_weights(pred_name)
            strat_model = hand_crafted_features_double_hinton()
            strat_model.load_weights(strat_name)
            players += [
                HandCraftEverywhereRnnPlayer(pred_model, strat_model, pred_mem, strat_mem, normal_pred_y_func,
                                             normal_strat_y_func, 0.001, 0.001, 1, 1, frozen=True)
                for _ in range(4)
            ]
        print("The networks from the first checkpoint have been added")


def many_players_magic(discount: int) \
        -> Tuple[List[DifferenzlerPlayer], List[Tuple[keras.Model, Memory, keras.Model, Memory, int, str]]]:
    # we want 2 normal player nets, 1 each for aggressive, defensive, hyper aggressive, hyper defensive
    memory_scaling = 6 * 2
    path_to_active_nets = './ongoing_nets/active_nets/'
    ongoing_nets = os.listdir(path_to_active_nets) if path.exists(path_to_active_nets) else []

    def get_tuple(pred_y_func, strat_y_func, pred_net_path=None, strat_net_path=None):
        if pred_net_path is not None:
            print("loading {}".format(pred_net_path))
            print("loading {}".format(strat_net_path))
        pred_net = prediction_resnet()
        if pred_net_path is not None:
            pred_net.load_weights(pred_net_path)
        strat_net = hand_crafted_features_double_hinton()
        if strat_net_path is not None:
            strat_net.load_weights(strat_net_path)
        return (
            pred_net, strat_net, ReplayMemory(2_000 * memory_scaling),
            RnnReplayMemory(16_000 * memory_scaling), pred_y_func, strat_y_func, 0.06, 0.06,
            batch_size_pred, batch_size_strat
        )

    def get_net_name(contain: List[str], not_contain: List[str]=[]) -> str:
        tmp = filter(lambda name: all(map(lambda c: c in name, contain)), ongoing_nets)
        net_name = list(filter(lambda name: not any(map(lambda c: c in name, not_contain)), tmp))
        if len(net_name) != 1:
            print("Loading ongoing nets failed. The given filters are:")
            print(contain, not_contain)
            print("The matches are: {}".format(net_name))
            exit()
        return path_to_active_nets + net_name[0]

    net_paths_ongoing = [(None, None)] * 6
    if len(ongoing_nets) != 0:
        net_paths_ongoing = [
            (get_net_name(['pred', "normal_1"]), get_net_name(['strat', 'normal_1'])),
            (get_net_name(['pred', "normal_2"]), get_net_name(['strat', 'normal_2'])),
            (get_net_name(['pred', "aggressive"], ['hyper']), get_net_name(['strat', 'aggressive'], ['hyper'])),
            (get_net_name(['pred', "defensive"], ['hyper']), get_net_name(['strat', 'defensive'], ['hyper'])),
            (get_net_name(['pred', "hyper_aggressive"]), get_net_name(['strat', 'hyper_aggressive'])),
            (get_net_name(['pred', "hyper_defensive"]), get_net_name(['strat', 'hyper_defensive'])),
        ]

    player_args = [
        get_tuple(normal_pred_y_func, normal_strat_y_func, *net_paths_ongoing[0]),
        get_tuple(normal_pred_y_func, normal_strat_y_func, *net_paths_ongoing[1]),
        get_tuple(normal_pred_y_func, aggressive_strat_y_func, *net_paths_ongoing[2]),
        get_tuple(normal_pred_y_func, defensive_strat_y_func, *net_paths_ongoing[3]),
        get_tuple(normal_pred_y_func, very_aggressive_strat_y_func, *net_paths_ongoing[4]),
        get_tuple(normal_pred_y_func, very_defensive_strat_y_func, *net_paths_ongoing[5]),
    ]

    players = sum([
        [
            HandCraftEverywhereRnnPlayer(*player_arg) for _ in range(4 * fit_window)
        ]
        for player_arg in player_args
    ], [])
    add_frozen_players(players)

    name_bases = [
        "double_hinton_normal_1_discount_{}_dropout_50_player".format(discount),
        "double_hinton_normal_2_discount_{}_dropout_50_player".format(discount),
        "double_hinton_aggressive_discount_{}_dropout_50_player".format(discount),
        "double_hinton_defensive_discount_{}_dropout_50_player".format(discount),
        "double_hinton_hyper_aggressive_discount_{}_dropout_50_player".format(discount),
        "double_hinton_hyper_defensive_discount_{}_dropout_50_player".format(discount),
    ]
    train_tuples = [
        (
            players[i]._prediction_model,
            players[i]._prediction_memory,
            players[i].strategy_model,
            players[i]._strategy_memory,
            1,  # the training factor
            name_bases[i // 4 // fit_window]
        ) for i in range(0, 4 * fit_window * len(name_bases), 4 * fit_window)
    ]
    return players, train_tuples


def main():
    discount = 4
    players, training_tuples = many_players_magic(discount)
    og_training_tuples_length = len(training_tuples)
    # print("\n\n\nCurrently training: {}".format(name_base))
    sitting = DifferenzlerSitting()
    sitting.set_players(players)
    training_start_time = datetime.datetime.now()
    rounds_played = 0
    if path.exists('./ongoing_nets/active_nets') and len('./ongoing_nets/active_nets') != 0:
        rounds_played = int(os.listdir('./ongoing_nets/active_nets')[0].split('_')[-1][:-3]) // fit_window
    print("The training begins with {} players and rounds_played = {}, at {}"
          .format(len(players), rounds_played, training_start_time))
    while (datetime.datetime.now() - training_start_time).total_seconds() < 7 * 3600:
        if rounds_played % 10 == 0:
            print(rounds_played, (datetime.datetime.now() - training_start_time).total_seconds())
        sitting.play_full_round(train=False, discount=discount, shuffle=True)
        for pred_model, pred_mem, strat_model, strat_mem, training_factor, _ in training_tuples:
            xs_pred, ys_pred = pred_mem.draw_batch(sample_limit_pred * training_factor)
            xs_strat, ys_strat = strat_mem.draw_batch(sample_limit_strat * training_factor)

            tmp = datetime.datetime.now()
            pred_model.fit(xs_pred, ys_pred, batch_size=batch_size_pred, verbose=0)
            strat_model.fit(xs_strat, ys_strat, batch_size=batch_size_strat, verbose=0)
            RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
            RnnPlayer.time_spent_training += datetime.datetime.now() - tmp

        rounds_played += 1

        if (rounds_played < 15_000 and rounds_played % 1_000 == 0) or rounds_played % 25_000 == 0:
            print("We passed {} rounds and it's {}".format(rounds_played, datetime.datetime.now()))
            print("time spent in keras = {}".format(RnnPlayer.total_time_spent_in_keras))
            print("time spent training = {}".format(RnnPlayer.time_spent_training))
            RnnPlayer.total_time_spent_in_keras = datetime.timedelta()
            RnnPlayer.time_spent_training = datetime.timedelta()

        if rounds_played == 8_000 // fit_window:  # freeze copies of the non-normal players
            freeze_players(players, og_training_tuples_length, rounds_played, training_tuples, 1, 2)
            print("The 1st freeze round has been performed. We have {} players now.".format(len(players)))
            sitting.set_players(players)
        elif rounds_played == 80_000 // fit_window:
            freeze_players(players, og_training_tuples_length, rounds_played, training_tuples, fit_window // 2)
            print("The 2nd freeze round has been performed. We have {} players now.".format(len(players)))
            sitting.set_players(players)
        elif rounds_played == 500_000 // fit_window:
            freeze_players(players, og_training_tuples_length, rounds_played, training_tuples, fit_window // 2)
            print("The 3rd freeze round has been performed. We have {} players now.".format(len(players)))
            sitting.set_players(players)

    save_current_nets(rounds_played, training_tuples)
    print("training is over with {} rounds played at {}".format(rounds_played, datetime.datetime.now()))


def save_current_nets(rounds_played, training_tuples):
    print("saving the current networks at rounds_played = {}".format(rounds_played))
    for pred_model, _, strat_model, _, _, name_base in training_tuples:
        pred_model.save("./pred_{}_{}.h5".format(name_base, rounds_played * fit_window))
        strat_model.save("./strat_{}_{}.h5".format(name_base, rounds_played * fit_window))


def freeze_players(
        players: List[DifferenzlerPlayer], og_lenght: int, rounds_played: int,
        training_tuples, nbr_of_tables: int, start_index: int = 0
):
    new_pred_mem = ReplayMemory(1)
    new_strat_mem = RnnReplayMemory(1)
    for k in range(start_index, og_lenght):
        pred_m, _, strat_m, _, _, name_b = training_tuples[k]
        pred_name = "./pred_{}_{}.h5".format(name_b, rounds_played * fit_window)
        strat_name = "./strat_{}_{}.h5".format(name_b, rounds_played * fit_window)
        pred_m.save(pred_name)
        strat_m.save(strat_name)
        new_pred_model = keras.models.load_model(pred_name)
        new_strat_model = keras.models.load_model(strat_name)
        players += [
            HandCraftEverywhereRnnPlayer(new_pred_model, new_strat_model, new_pred_mem, new_strat_mem,
                                         normal_pred_y_func, normal_strat_y_func, 0.001, 0.001, 1, 1, frozen=True)
            for _ in range(4 * nbr_of_tables)
        ]


if __name__ == '__main__':
    main()
