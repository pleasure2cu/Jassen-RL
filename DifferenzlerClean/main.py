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
    hand_crafted_features_double_hinton
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer, HandCraftEverywhereRnnPlayer
from sitting import DifferenzlerSitting

number_of_epochs = 4  # decides how many times the intermediate stats are written
epoch_size = 25_000  # decides over how many rounds an intermediate stats text goes
fit_window = 15  # after how many rounds the model is trained
sample_coverage = 1.0  # what percentage of samples do you want to be looked at (in the optimal case)
batch_size_strat = 192
sample_limit_strat = int(6 * 32 * fit_window * sample_coverage / batch_size_strat + 1) * batch_size_strat
batch_size_pred = int(batch_size_strat / 8 + 1)
sample_limit_pred = int(6 * 4 * fit_window * sample_coverage / batch_size_pred + 1) * batch_size_pred
discount = 32
print("Batch size for strat = {}".format(batch_size_strat))
print("Sample limit strategy = {}".format(sample_limit_strat))


def some_magic() -> Tuple[List[DifferenzlerPlayer], List[Tuple[keras.Model, Memory, keras.Model, Memory, int]], str]:
    dropout = 0.5

    pred_memory = ReplayMemory(2_000 * 6)
    strat_memory = RnnReplayMemory(16_000 * 6)

    pred_model: keras.Model = prediction_resnet()
    strat_model = hand_crafted_features_hinton(dropout=0.5)
    strat_model.summary()

    players = [
        HandCraftEverywhereRnnPlayer(
            pred_model, strat_model, pred_memory, strat_memory,
            normal_pred_y_func, normal_strat_y_func, 0.07, 0.07, batch_size_pred, batch_size_strat
        )
        for _ in range(4 * fit_window)
    ]

    return players, [(pred_model, pred_memory, strat_model, strat_memory, 1)], \
           "fourth_reproduce_hinton_net_{}_discount_{}_dropout_player".format(discount, int(dropout * 100))


def main():
    players, training_tuples, name_base = some_magic()
    print("\n\n\nCurrently training: {}".format(name_base))
    sitting = DifferenzlerSitting()
    sitting.set_players(players)
    training_start_time = datetime.datetime.now()
    for epoch_index in range(number_of_epochs):
        epoch_start_time = datetime.datetime.now()
        total_diff = 0
        for i in range(0, epoch_size, fit_window):
            diffs = sitting.play_full_round(train=False, discount=discount, shuffle=False)
            total_diff += np.sum(diffs)
            for pred_model, pred_mem, strat_model, strat_mem, training_factor in training_tuples:
                xs_pred, ys_pred = pred_mem.draw_batch(sample_limit_pred * training_factor)
                xs_strat, ys_strat = strat_mem.draw_batch(sample_limit_strat * training_factor)

                tmp = datetime.datetime.now()
                pred_model.fit(xs_pred, ys_pred, batch_size=batch_size_pred, verbose=0)
                strat_model.fit(xs_strat, ys_strat, batch_size=batch_size_strat, verbose=0)
                RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
                RnnPlayer.time_spent_training += datetime.datetime.now() - tmp

        print("\ntime spent in total = {}".format(datetime.datetime.now() - epoch_start_time))
        print("time spent in keras = {}".format(RnnPlayer.total_time_spent_in_keras))
        print("time spent training = {}".format(RnnPlayer.time_spent_training))
        print("avg diff = {}".format(total_diff / epoch_size / 4))
        RnnPlayer.total_time_spent_in_keras = datetime.timedelta()
        RnnPlayer.time_spent_training = datetime.timedelta()
        print(
            "estimated finish time:",
            training_start_time + (datetime.datetime.now() - training_start_time) / (epoch_index + 1) * number_of_epochs
        )

    for pred_model, _, strat_model, _, _ in training_tuples:
        pred_model.save("./pred_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))
        strat_model.save("./strat_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))


if __name__ == '__main__':
    main()
