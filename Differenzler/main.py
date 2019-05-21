import datetime
import random

import numpy as np
from keras.models import load_model

from Memory import ReplayMemory, RnnReplayMemory
from Network import PredictionNetwork, RnnStrategyNetwork
from Player import RnnPlayer
from PlayerInterlayer import RnnPlayerInterlayer
from Sitting import Sitting
from main_helper_methods import normal_pred_y_func, normal_strat_y_func, \
    aggressive_strat_y_func, defensive_strat_y_func, small_prediction_net, very_aggressive_strat_y_func, \
    very_defensive_strat_y_func, small_rnn_strategy

prediction_save_path = './saved_nets/prediction/'
strategy_save_path = './saved_nets/strategy/'

# Parameters
prediction_replay_memory_size = 1500
strategy_replay_memory_size = 1500

prediction_net_batch_size = 64
strategy_net_batch_size = 128

prediction_exploration_rate = 0.07
strategy_exploration_rate = 0.07

size_of_one_strat_net_input = 83

total_rounds = 8000000
rounds_until_save = 100000
interval_to_print_stats = 25000
round_when_adding_players = 1000000
start_offset = 0

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

if round_when_adding_players % rounds_until_save != 0:
    assert 0, "round_when_adding_players isn't a multiple of rounds_until_save"


def main():
    # create replay memories
    pred_memories = [ReplayMemory(prediction_replay_memory_size) for _ in range(5)]
    strat_memories = [RnnReplayMemory(strategy_replay_memory_size) for _ in range(5)]

    # create Networks
    pred_networks = [
        PredictionNetwork(small_prediction_net(), pred_memories[0], prediction_net_batch_size, True),
        PredictionNetwork(small_prediction_net(), pred_memories[1], prediction_net_batch_size, True),
        PredictionNetwork(small_prediction_net(), pred_memories[2], prediction_net_batch_size, True),
        PredictionNetwork(small_prediction_net(), pred_memories[3], prediction_net_batch_size, True),
        PredictionNetwork(small_prediction_net(), pred_memories[4], prediction_net_batch_size, True)
    ]
    print(pred_networks[0]._neural_network.summary())
    strat_networks = [
        RnnStrategyNetwork(small_rnn_strategy(), strat_memories[0], strategy_net_batch_size, True),
        RnnStrategyNetwork(small_rnn_strategy(), strat_memories[1], strategy_net_batch_size, True),
        RnnStrategyNetwork(small_rnn_strategy(), strat_memories[2], strategy_net_batch_size, True),
        RnnStrategyNetwork(small_rnn_strategy(), strat_memories[3], strategy_net_batch_size, True),
        RnnStrategyNetwork(small_rnn_strategy(), strat_memories[4], strategy_net_batch_size, True)
    ]
    print(strat_networks[0]._neural_network.summary())

    # make pairs of the networks
    networks = list(sum(zip(pred_networks, strat_networks), ()))

    # give each network a name
    pred_network_names = [
        'wide_rnn_very_aggressive_prediction',
        'wide_rnn_aggressive_prediction',
        'wide_rnn_very_defensive_prediction',
        'wide_rnn_defensive_prediction',
        'wide_rnn_normal_prediction'
    ]
    strat_network_names = [
        'wide_rnn_very_aggressive_strategy',
        'wide_rnn_aggressive_strategy',
        'wide_rnn_very_defensive_strategy',
        'wide_rnn_defensive_strategy',
        'wide_rnn_normal_strategy'
    ]

    # make the same pairs as above
    network_names = list(sum(zip(pred_network_names, strat_network_names), ()))

    # create players
    players = [
        [RnnPlayer(pred_networks[0], strat_networks[0], prediction_exploration_rate, strategy_exploration_rate)],
        [RnnPlayer(pred_networks[1], strat_networks[1], prediction_exploration_rate, strategy_exploration_rate)
         for _ in range(2)],
        [RnnPlayer(pred_networks[2], strat_networks[2], prediction_exploration_rate, strategy_exploration_rate)],
        [RnnPlayer(pred_networks[3], strat_networks[3], prediction_exploration_rate, strategy_exploration_rate)
         for _ in range(2)],
        [RnnPlayer(pred_networks[4], strat_networks[4], prediction_exploration_rate, strategy_exploration_rate)
         for _ in range(4)]
    ]

    # flatten players
    players = sum(players, [])

    # create one PlayerInterlayer for each player
    players = [
        [RnnPlayerInterlayer(player, normal_pred_y_func, very_aggressive_strat_y_func) for player in players[: 1]],
        [RnnPlayerInterlayer(player, normal_pred_y_func, aggressive_strat_y_func) for player in players[1: 3]],
        [RnnPlayerInterlayer(player, normal_pred_y_func, very_defensive_strat_y_func) for player in players[3: 4]],
        [RnnPlayerInterlayer(player, normal_pred_y_func, defensive_strat_y_func) for player in players[4: 6]],
        [RnnPlayerInterlayer(player, normal_pred_y_func, normal_strat_y_func) for player in players[6:]]
    ]
    players = sum(players, [])

    # create one Sitting
    sitting = Sitting(debugging)
    last_stop = datetime.datetime.now()
    r = random.Random()
    with open('stats_dev.txt', 'w') as f:
        f.write("// interval to print stats: " + str(interval_to_print_stats) + "\n")
        total_diff = 0
        total_losses = [0.0 for _ in range(len(networks))]
        for i in range(start_offset, total_rounds, 10):
            sitting.set_players(r.sample(players, 4))
            for _ in range(10):
                total_diff += sitting.play_full_round()
            i += 9
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
            if i + 1 == round_when_adding_players:
                print('adding players')
                # add 2 more normal players
                nps = [RnnPlayer(networks[-2], networks[-1], prediction_exploration_rate, strategy_exploration_rate)
                       for _ in range(2)]
                inps = [RnnPlayerInterlayer(nps[i], normal_pred_y_func, normal_strat_y_func) for i in range(2)]
                players += inps

                # add 2 static versions of the current normal player
                pred_mem = ReplayMemory(1)
                strat_mem = RnnReplayMemory(1)
                pred_net = load_model(prediction_save_path + 'normal_prediction_' + str(i + 1) + '.h5')
                strat_net = load_model(strategy_save_path + 'normal_strategy_' + str(i + 1) + '.h5')
                p_net = PredictionNetwork(pred_net, pred_mem, 1, False)
                s_net = RnnStrategyNetwork(strat_net, strat_mem, 1, False)
                ps = [RnnPlayer(p_net, s_net, 0.02, 0.02) for _ in range(2)]
                ips = [RnnPlayerInterlayer(ps[i], normal_pred_y_func, normal_strat_y_func) for i in range(2)]
                players += ips


if __name__ == "__main__":
    main()
