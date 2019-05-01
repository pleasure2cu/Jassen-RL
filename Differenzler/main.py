import datetime
import random

import numpy as np

from Memory import ReplayMemory, RnnReplayMemory, MultiReplayMemory
from Network import PredictionNetwork, StrategyNetwork, RnnStrategyNetwork, MultiPredictionNetwork
from Player import Player, RnnPlayer
from PlayerInterlayer import PlayerInterlayer, RnnPlayerInterlayer, RnnMultiPlayerInterlayer
from Sitting import Sitting
from main_helper_methods import prediction_resnet, strategy_rnn_resnet, normal_pred_y_func, normal_strat_y_func

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
random.seed(42)

prediction_save_path = './saved_nets/prediction/prediction'
strategy_save_path = './saved_nets/strategy/strategy'

# Parameters
prediction_replay_memory_size = 100000
strategy_replay_memory_size = 20000

prediction_net_batch_size = 64
strategy_net_batch_size = 128

prediction_exploration_rate = 0.07
strategy_exploration_rate = 0.07

size_of_one_strat_net_input = 83

total_rounds = 1
rounds_until_save = 50000
interval_to_print_stats = 1

func_for_pred_y = normal_pred_y_func
func_for_strat_y = normal_strat_y_func

only_train_in_turn = False
turn_size = 2

use_batch_norm = True
debugging = False


if debugging and total_rounds > 10000:
    print("WARNING: you are still debugging")

if only_train_in_turn and (turn_size < rounds_until_save and rounds_until_save % turn_size != 0 or
                           rounds_until_save < turn_size and turn_size % rounds_until_save != 0):
    print("WARNING: turn_size (" + str(turn_size) + ") and rounds_until_save (" + str(rounds_until_save) + ") aren't "
                               "multiple of each other")

if total_rounds < rounds_until_save:
    rounds_until_save = total_rounds


def main():
    # create one ReplayMemory for each network kind
    pred_memory = ReplayMemory(prediction_replay_memory_size)
    strat_memory = RnnReplayMemory(strategy_replay_memory_size)

    # create one Network pair
    pred_network = PredictionNetwork(prediction_resnet(), pred_memory, prediction_net_batch_size, True)
    strat_network = RnnStrategyNetwork(strategy_rnn_resnet(use_batch_norm), strat_memory, strategy_net_batch_size, True)
    networks = [pred_network, strat_network]

    # create 4 players, each with the same Networks
    players = [RnnPlayer(pred_network, strat_network, prediction_exploration_rate, strategy_exploration_rate)
               for _ in range(4)]

    # create one PlayerInterlayer for each player
    players = [RnnPlayerInterlayer(players[i], i, func_for_pred_y, func_for_strat_y) for i in range(4)]

    # create one Sitting
    sitting = Sitting(debugging)
    last_stop = datetime.datetime.now()
    with open('stats.txt', 'w') as f:
        f.write("// interval to print stats: " + str(interval_to_print_stats) + "\n")
        total_diff = 0
        total_losses = [0.0 for _ in range(len(networks))]
        for i in range(total_rounds // rounds_until_save):
            for j in range(rounds_until_save):
                sitting.set_players(players)
                total_diff += sitting.play_full_round()
                for _ in range(1):  # just so that we actually learn a few times
                    if only_train_in_turn:
                        index_to_train = (i * total_rounds + j) // turn_size % len(networks)
                        total_losses[index_to_train] += networks[index_to_train].train()
                    else:
                        for net_i, network in enumerate(networks):
                            total_losses[net_i] += network.train()
                if (i * rounds_until_save + j + 1) % interval_to_print_stats == 0:
                    print(str(i * rounds_until_save + j + 1), "rounds have been played")
                    avg = total_diff / 4 / interval_to_print_stats
                    print("Average difference of one player:\t", avg)
                    losses_string = ', '.join([str(l) for l in np.array(total_losses) / interval_to_print_stats])
                    print("The losses are:\t", losses_string)
                    # print("It took:", datetime.datetime.now() - last_stop)
                    last_stop = datetime.datetime.now()
                    print('')
                    f.write(str(i * rounds_until_save + j + 1) + "\n")
                    f.write(str(avg) + "\n")
                    f.write(losses_string + "\n")
                    total_diff = 0
                    total_losses = [0.0 for _ in range(len(networks))]

            pred_network.save_network(prediction_save_path + '_' + str((i + 1) * rounds_until_save) + '.h5')
            strat_network.save_network(strategy_save_path + '_' + str((i + 1) * rounds_until_save) + '.h5')


if __name__ == "__main__":
    main()
