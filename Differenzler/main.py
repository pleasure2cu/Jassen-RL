import datetime
import random

import numpy as np

from Memory import ReplayMemory, RnnReplayMemory, MultiReplayMemory
from Network import PredictionNetwork, StrategyNetwork, RnnStrategyNetwork, MultiPredictionNetwork
from Player import Player, RnnPlayer
from PlayerInterlayer import PlayerInterlayer, RnnPlayerInterlayer, RnnMultiPlayerInterlayer
from Sitting import Sitting
from main_helper_methods import prediction_resnet, strategy_rnn_resnet, normal_pred_y_func, normal_strat_y_func, \
    aggressive_strat_y_func, defensive_strat_y_func, small_prediction_net, small_strategy_net

prediction_save_path = './saved_nets/prediction/'
strategy_save_path = './saved_nets/strategy/'

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
    # create replay memories
    pred_memories = [ReplayMemory(prediction_replay_memory_size) for _ in range(3)]
    pred_memories.append(ReplayMemory(1))
    strat_memories = [RnnReplayMemory(strategy_replay_memory_size) for _ in range(3)]
    strat_memories.append(RnnReplayMemory(1))

    # create Networks
    pred_networks = [
        PredictionNetwork(load_model('saved_nets/prediction_500000.h5'), pred_memories[0], prediction_net_batch_size, True),
        PredictionNetwork(load_model('saved_nets/prediction_500000.h5'), pred_memories[1], prediction_net_batch_size, True),
        PredictionNetwork(load_model('saved_nets/prediction_500000.h5'), pred_memories[2], prediction_net_batch_size, True),
        PredictionNetwork(small_prediction_net(), pred_memories[3], prediction_net_batch_size, False)
    ]
    strat_networks = [
        RnnStrategyNetwork(load_model('saved_nets/strategy_500000.h5'), strat_memories[0], strategy_net_batch_size, True),
        RnnStrategyNetwork(load_model('saved_nets/strategy_500000.h5'), strat_memories[1], strategy_net_batch_size, True),
        RnnStrategyNetwork(load_model('saved_nets/strategy_500000.h5'), strat_memories[2], strategy_net_batch_size, True),
        RnnStrategyNetwork(small_strategy_net(), strat_memories[3], strategy_net_batch_size, False)
    ]

    # make pairs of the networks
    networks = list(sum(zip(pred_networks, strat_networks), ()))

    # give each network a name
    pred_network_names = [
        'normal_prediction',
        'aggressive_prediction',
        'defensive_prediction',
        'random_prediction',
    ]
    strat_network_names = [
        'normal_strategy',
        'aggressive_strategy',
        'defensive_strategy',
        'random_strategy'
    ]

    # make the same pairs as above
    network_names = list(sum(zip(pred_network_names, strat_network_names), ()))

    # create players
    players = [
        [RnnPlayer(pred_networks[0], strat_networks[0], prediction_exploration_rate, strategy_exploration_rate)
         for _ in range(4)],
        [RnnPlayer(pred_networks[1], strat_networks[1], prediction_exploration_rate, strategy_exploration_rate)
         for _ in range(3)],
        [RnnPlayer(pred_networks[2], strat_networks[2], prediction_exploration_rate, strategy_exploration_rate)
         for _ in range(3)],
        [RnnPlayer(pred_networks[3], strat_networks[3], 0.99999, 0.99999)
         for _ in range(2)]
    ]

    # flatten players
    players = sum(players, [])

    # create one PlayerInterlayer for each player
    players = [RnnPlayerInterlayer(player, func_for_pred_y, func_for_strat_y) for player in players]

    # create one Sitting
    sitting = Sitting(debugging)
    last_stop = datetime.datetime.now()
    r = random.Random()
    with open('stats.txt', 'w') as f:
        f.write("// interval to print stats: " + str(interval_to_print_stats) + "\n")
        total_diff = 0
        total_losses = [0.0 for _ in range(len(networks))]
        for i in range(total_rounds):
            sitting.set_players(r.sample(players, 4))
            total_diff += sitting.play_full_round()
            if only_train_in_turn:
                index_to_train = i // turn_size % len(networks)
                total_losses[index_to_train] += networks[index_to_train].train()
            else:
                for net_i, network in enumerate(networks):
                    total_losses[net_i] += network.train()
            if (i + 1) % interval_to_print_stats == 0:
                print(str(i + 1), "rounds have been played")
                avg = total_diff / 4 / interval_to_print_stats
                print("Average difference of one player:\t", avg)
                losses_string = ', '.join([str(l) for l in np.array(total_losses) / interval_to_print_stats])
                print("The losses are:\t", losses_string)
                print("It took:", datetime.datetime.now() - last_stop)
                last_stop = datetime.datetime.now()
                print('')
                f.write(str(i + 1) + "\n")
                f.write(str(avg) + "\n")
                f.write(losses_string + "\n")
                total_diff = 0
                total_losses = [0.0 for _ in range(len(networks))]
            if (i + 1) % rounds_until_save == 0:
                for keras_net, net_name in zip(networks, network_names):
                    if 'random' in net_name:
                        continue
                    elif 'pred' in net_name:
                        full_name = prediction_save_path
                    elif 'strat' in net_name:
                        full_name = strategy_save_path
                    else:
                        assert 0, net_name
                    full_name += net_name + '_' + str(i + 1) + '.h5'
                    keras_net.save_network(full_name)


if __name__ == "__main__":
    main()
