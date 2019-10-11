import datetime

import numpy as np

from main_helper_methods import prediction_resnet, strategy_deep_lstm_resnet, normal_pred_y_func, normal_strat_y_func, \
    prediction_l1_resnet, normal_strategy_network, small_strategy_network, tiny_strategy_network, \
    small_rnn_strategy_network, small_l1_strategy_network
from memory import ReplayMemory, RnnReplayMemory
from player import RnnPlayer
from sitting import DifferenzlerSitting


number_of_epochs = 5
epoch_size = 300
batch_size = 192
fit_window = 12


def main():
    pred_model_funcs = [prediction_resnet]
    strat_model_funcs = [small_strategy_network]
    name_bases = ["small_player_2nd_edition"]

    for pred_model_func, strat_model_func, name_base in zip(pred_model_funcs, strat_model_funcs, name_bases):

        print("\n\n\nCurrently training: {}".format(name_base))

        pred_memory = ReplayMemory(1_000*6)
        strat_memory = RnnReplayMemory(9_000*6)

        pred_model = pred_model_func()
        strat_model = strat_model_func()

        players = [
            RnnPlayer(
                pred_model, strat_model, pred_memory, strat_memory,
                normal_pred_y_func, normal_strat_y_func, 0.07, 0.07, batch_size
            )
            for _ in range(4 * 12)
        ]

        sitting = DifferenzlerSitting()
        sitting.set_players(players)
        for epoch_index in range(number_of_epochs):
            epoch_start_time = datetime.datetime.now()
            total_diff = 0
            total_loss_p = 0.
            total_loss_s = 0.
            for i in range(0, epoch_size, 12):
                # print("{}".format(epoch_index*epoch_size+i), end='\r')
                loss_p, loss_s, diffs = sitting.play_full_round(train=False, nbr_of_parallel_rounds=12)
                total_diff += np.sum(diffs)
                total_loss_p += loss_p
                total_loss_s += loss_s
                assert pred_memory.assert_items()
                assert strat_memory.assert_items()
                if i % fit_window == 0:
                    sample_limit = 4 * 4 * batch_size

                    xs_pred, ys_pred = pred_memory.draw_batch(sample_limit)
                    xs_strat, ys_strat = strat_memory.draw_batch(sample_limit)

                    tmp = datetime.datetime.now()
                    pred_model.fit(xs_pred, ys_pred, batch_size=batch_size, verbose=0)
                    strat_model.fit(xs_strat, ys_strat, batch_size=batch_size, verbose=0)
                    RnnPlayer.total_time_spent_in_keras += datetime.datetime.now() - tmp
                    RnnPlayer.time_spent_training += datetime.datetime.now() - tmp
            print(datetime.datetime.now() - epoch_start_time)
            print("avg diff = {} \t loss_p = {} \t loss_s = {}".format(total_diff/epoch_size/4*12, total_loss_p, total_loss_s))
            print("time spent in keras = {}".format(RnnPlayer.total_time_spent_in_keras))
            print("time spent training = {}".format(RnnPlayer.time_spent_training))
            RnnPlayer.total_time_spent_in_keras = datetime.timedelta()
            RnnPlayer.time_spent_training = datetime.timedelta()

        pred_model.save("./pred_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))
        strat_model.save("./strat_{}_{}.h5".format(name_base, number_of_epochs * epoch_size))


if __name__ == '__main__':
    main()
