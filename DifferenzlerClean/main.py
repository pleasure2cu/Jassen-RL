import datetime
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


def many_players_magic(discount: int) \
        -> Tuple[List[DifferenzlerPlayer], List[Tuple[keras.Model, Memory, keras.Model, Memory, int, str]]]:
    # we want 2 normal player nets, 1 each for aggressive, defensive, hyper aggressive, hyper defensive
    memory_scaling = 6 * 2

    def get_tuple(pred_y_func, strat_y_func):
        return (
            prediction_resnet(), hand_crafted_features_double_hinton(), ReplayMemory(2_000 * memory_scaling),
            RnnReplayMemory(16_000 * memory_scaling), pred_y_func, strat_y_func, 0.06, 0.06,
            batch_size_pred, batch_size_strat
        )

    player_args = [
        get_tuple(normal_pred_y_func, normal_strat_y_func),
        get_tuple(normal_pred_y_func, normal_strat_y_func),
        get_tuple(normal_pred_y_func, aggressive_strat_y_func),
        get_tuple(normal_pred_y_func, defensive_strat_y_func),
        get_tuple(normal_pred_y_func, very_aggressive_strat_y_func),
        get_tuple(normal_pred_y_func, very_defensive_strat_y_func),
    ]

    players = sum([
        [
            HandCraftEverywhereRnnPlayer(*player_arg) for _ in range(4 * fit_window)
        ]
        for player_arg in player_args
    ], [])

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
        ) for i in range(0, len(players), 4 * fit_window)
    ]
    return players, train_tuples


def main():
    discount = 4
    players, training_tuples = many_players_magic(discount)
    players_og_length = len(players)
    # print("\n\n\nCurrently training: {}".format(name_base))
    sitting = DifferenzlerSitting()
    sitting.set_players(players)
    training_start_time = datetime.datetime.now()
    print("The training begins ({})".format(training_start_time))
    rounds_played = 0
    while (datetime.datetime.now() - training_start_time).total_seconds() < 11 * 3600:
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

        if (rounds_played < 26_000 and rounds_played % 5_000 == 0) or rounds_played % 25_000 == 0:
            print("We passed {} rounds and it's {}".format(rounds_played, datetime.datetime.now()))
            print("time spent in keras = {}".format(RnnPlayer.total_time_spent_in_keras))
            print("time spent training = {}".format(RnnPlayer.time_spent_training))
            RnnPlayer.total_time_spent_in_keras = datetime.timedelta()
            RnnPlayer.time_spent_training = datetime.timedelta()

        if rounds_played == 8_000 // fit_window:  # freeze copies of the non-normal players
            freeze_players(players, players_og_length, rounds_played, training_tuples, 1, 2 * 4 * fit_window)
            print("The 1st freeze round has been performed. We have {} players now.".format(len(players)))
        elif rounds_played == 80_000 // fit_window:
            freeze_players(players, players_og_length, rounds_played, training_tuples, fit_window // 2)
            print("The 2nd freeze round has been performed. We have {} players now.".format(len(players)))
        elif rounds_played == 500_000 // fit_window:
            freeze_players(players, players_og_length, rounds_played, training_tuples, fit_window // 2)
            print("The 3rd freeze round has been performed. We have {} players now.".format(len(players)))

    for pred_model, _, strat_model, _, _, name_base in training_tuples:
        pred_model.save("./pred_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))
        strat_model.save("./strat_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))
    print("training is over")


def freeze_players(
        players: List[DifferenzlerPlayer], og_lenght: int, rounds_played: int,
        training_tuples, nbr_of_tables: int, start_index: int = 0
):
    new_pred_mem = ReplayMemory(1)
    new_strat_mem = RnnReplayMemory(1)
    for k in range(start_index, og_lenght, 4 * fit_window):
        pred_m, _, strat_m, _, _, name_b = training_tuples[k]
        pred_name = "./pred_{}_{}.h5".format(name_b, rounds_played * fit_window)
        strat_name = "./strat_{}_{}.h5".format(name_b, rounds_played * fit_window)
        pred_m.save(pred_name)
        strat_m.save(strat_name)
        new_pred_model = keras.models.load_model(pred_name)
        new_strat_model = keras.models.load_model(strat_name)
        players += [
            HandCraftEverywhereRnnPlayer(new_pred_model, new_strat_model, new_pred_mem, new_strat_mem,
                                         normal_pred_y_func, normal_strat_y_func, 0.001, 0.001, 1, 1)
            for _ in range(4 * nbr_of_tables)
        ]


if __name__ == '__main__':
    main()
