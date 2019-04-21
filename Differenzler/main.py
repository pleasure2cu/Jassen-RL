import keras
from keras import Model, Input
from keras.layers import Dense, BatchNormalization, np, Activation, LSTM

from Memory import ReplayMemory, RnnReplayMemory
from Network import PredictionNetwork, StrategyNetwork, RnnStrategyNetwork
from Player import Player, RnnPlayer
from PlayerInterlayer import PlayerInterlayer, RnnPlayerInterlayer
from Sitting import Sitting
from helper_functions import resnet_block

prediction_save_path = './saved_nets/prediction/prediction'
strategy_save_path = './saved_nets/strategy/strategy'

# Parameters
prediction_replay_memory_size = 36 * 200
strategy_replay_memory_size = 4 * 200

prediction_net_batch_size = 64
strategy_net_batch_size = 128

prediction_exploration_rate = 0.07
strategy_exploration_rate = 0.07

size_of_one_strat_net_input = 83

total_rounds = 30000
rounds_until_save = 10000
interval_to_print_stats = 1000

use_batch_norm = True


def prediction_vanilla_ffn():
    net = keras.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=(37,)),
        keras.layers.Dense(30, activation='relu'),
        keras.layers.Dense(1)
    ])
    net.compile(optimizer='rmsprop', loss='mse')
    return net


def strategy_vanilla_ffn():
    net = keras.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=(size_of_one_strat_net_input,)),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(1)
    ])
    net.compile(optimizer='rmsprop', loss='mse')
    return net


def prediction_resnet():
    dense_output_size = 50
    net_input = Input(shape=(37,))
    layer_1 = Dense(dense_output_size, activation='relu')(net_input)
    layer_2 = Dense(dense_output_size, activation='relu')(layer_1)
    layer_3 = Dense(dense_output_size, activation='relu')(layer_2)
    res_sum = keras.layers.add([layer_1, layer_3])
    final_tensor = Dense(1)(res_sum)
    model = Model(inputs=net_input, outputs=final_tensor)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def strategy_resnet():
    dense_output_size = 120
    net_input = Input(shape=(size_of_one_strat_net_input,))
    net = Dense(dense_output_size)(net_input)
    if use_batch_norm:
        net = BatchNormalization()(net)
    net = Activation('relu')(net)
    for _ in range(3):
        net = resnet_block(net, dense_output_size, use_batch_norm)
    final_tensor = Dense(1)(net)
    model = Model(inputs=net_input, outputs=final_tensor)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def strategy_rnn_resnet():
    dense_output_size = 120
    rnn_output_size = 32
    rnn_input = Input(shape=(None, 9))
    rnn_output = LSTM(rnn_output_size)(rnn_input)
    aux_input = Input(shape=(87,))
    concat = keras.layers.concatenate([rnn_output, aux_input])
    net = Dense(dense_output_size)(concat)
    if use_batch_norm:
        net = BatchNormalization()(net)
    net = Activation('relu')(net)
    for _ in range(5):
        net = resnet_block(net, dense_output_size, use_batch_norm)
    final_tensor = Dense(1)(net)
    model = Model(inputs=[rnn_input, aux_input], outputs=final_tensor)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def main():
    # create one ReplayMemory for each network kind
    pred_memory = ReplayMemory(prediction_replay_memory_size)
    strat_memory = RnnReplayMemory(strategy_replay_memory_size)

    # create one Network pair
    pred_network = PredictionNetwork(prediction_resnet(), pred_memory, prediction_net_batch_size)
    strat_network = RnnStrategyNetwork(strategy_rnn_resnet(), strat_memory, strategy_net_batch_size)
    networks = [pred_network, strat_network]

    # create 4 players, each with the same Networks
    players = [RnnPlayer(pred_network, strat_network, prediction_exploration_rate, strategy_exploration_rate)
               for _ in range(4)]

    # create one PlayerInterlayer for each player
    players = [RnnPlayerInterlayer(players[i], i) for i in range(4)]

    # create one Sitting
    sitting = Sitting(players)
    with open('stats.txt', 'w') as f:
        for i in range(total_rounds // rounds_until_save):
            total_diff = 0
            for j in range(rounds_until_save):
                total_diff += sitting.play_full_round()
                for network in networks:
                    network.train()
                if (i * rounds_until_save + j + 1) % interval_to_print_stats == 0:
                    print(str(i * rounds_until_save + j + 1), "rounds have been played")
                    avg = total_diff / 4 / interval_to_print_stats
                    print("Average difference of one player:\t", avg)
                    print('')
                    f.write(str(avg) + "\n")
                    total_diff = 0

            pred_network.save_network(prediction_save_path + '_' + str((i + 1) * rounds_until_save) + '.h5')
            strat_network.save_network(strategy_save_path + '_' + str((i + 1) * rounds_until_save) + '.h5')


if __name__ == "__main__":
    main()
    prediction_resnet().summary()
    strategy_resnet().summary()
