import datetime
from typing import List

import numpy as np

from main_helper_methods import prediction_resnet, strategy_deep_lstm_resnet, normal_pred_y_func, normal_strat_y_func, \
    normal_strategy_network, small_strategy_network, tiny_strategy_network, \
    small_rnn_strategy_network, hand_crafted_features_rnn_network, \
    hand_crafted_features_rnn_network_wider, small_bidirectional_strategy_network, hand_crafted_features_hinton, \
    hand_crafted_features_double_hinton, hand_crafted_features_quad_hinton
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer, HandCraftEverywhereRnnPlayer
from sitting import DifferenzlerSitting


number_of_epochs = 8  # decides how many times the intermediate stats are written
epoch_size = 25_000  # decides over how many rounds an intermediate stats text goes
fit_window = 15  # after how many rounds the model is trained
parallel_rounds = fit_window
sample_coverage = 1.0  # what percentage of samples do you want to be looked at (in the optimal case)
batch_size_strat = 192
sample_limit_strat = int(6 * 32 * fit_window * sample_coverage / batch_size_strat + 1) * batch_size_strat
batch_size_pred = int(batch_size_strat / 8 + 1)
sample_limit_pred = int(6 * 4 * fit_window * sample_coverage / batch_size_pred + 1) * batch_size_pred
print("Batch size for strat = {}".format(batch_size_strat))
print("Sample limit strategy = {}".format(sample_limit_strat))

if fit_window % parallel_rounds != 0:
    print("fit_window is not a multiple of parallel_rounds, so the system won't train.")
    exit()


def main():
    for discount, dropout in zip([96], [0.5]):
        pred_model_funcs = [prediction_resnet]
        strat_model_funcs = [hand_crafted_features_quad_hinton]
        name_bases = ["quad_hinton_net_{}_discount_{}_dropout_player".format(discount, int(dropout*100))]

        for pred_model_func, strat_model_func, name_base in zip(pred_model_funcs, strat_model_funcs, name_bases):

            print("\n\n\nCurrently training: {}".format(name_base))

            pred_memory = ReplayMemory(2_000 * 6)
            strat_memory = RnnReplayMemory(16_000 * 6)

            pred_model = pred_model_func()
            strat_model = strat_model_func(dropout=dropout)
            print(strat_model.summary())

            players = [
                HandCraftEverywhereRnnPlayer(
                    pred_model, strat_model, pred_memory, strat_memory,
                    normal_pred_y_func, normal_strat_y_func, 0.07, 0.07, batch_size_pred, batch_size_strat
                )
                for _ in range(4 * parallel_rounds)
            ]

            sitting = DifferenzlerSitting()
            sitting.set_players(players)
            for epoch_index in range(number_of_epochs):
                epoch_start_time = datetime.datetime.now()
                total_diff = 0
                for i in range(0, epoch_size, parallel_rounds):
                    # print("{}".format(epoch_index*epoch_size+i), end='\r')
                    diffs = sitting.play_full_round(
                        train=parallel_rounds == 1 and i % fit_window == 0,
                        nbr_of_parallel_rounds=parallel_rounds,
                        strategy_model=strat_model,
                        discount=discount
                    )
                    total_diff += np.sum(diffs)
                    assert pred_memory.assert_items()
                    assert strat_memory.assert_items()
                    if i % fit_window == 0 and parallel_rounds > 1:
                        xs_pred, ys_pred = pred_memory.draw_batch(sample_limit_pred)
                        xs_strat, ys_strat = strat_memory.draw_batch(sample_limit_strat)

                        tmp = datetime.datetime.now()
                        pred_model.fit(xs_pred, ys_pred, batch_size=batch_size_pred, verbose=0)
                        strat_model.fit(xs_strat, ys_strat, batch_size=batch_size_strat, verbose=0)
                        RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
                        RnnPlayer.time_spent_training += datetime.datetime.now() - tmp
                print("\ntime spent in total = {}".format(datetime.datetime.now() - epoch_start_time))
                print("time spent in keras = {}".format(RnnPlayer.total_time_spent_in_keras))
                print("time spent training = {}".format(RnnPlayer.time_spent_training))
                print("avg diff = {}".format(total_diff/epoch_size/4))
                RnnPlayer.total_time_spent_in_keras = datetime.timedelta()
                RnnPlayer.time_spent_training = datetime.timedelta()

                # if (epoch_index + 1) % 3 == 0:
                #     pred_model.save("./pred_{}_{}.h5".format(name_base, (epoch_index + 1) * epoch_size))
                #     strat_model.save("./strat_{}_{}.h5".format(name_base, (epoch_index + 1) * epoch_size))

            pred_model.save("./pred_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))
            strat_model.save("./strat_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))


if __name__ == '__main__':
    main()
