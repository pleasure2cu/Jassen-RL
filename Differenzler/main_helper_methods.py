import keras
import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Dense, Activation, BatchNormalization

from helper_functions import resnet_block


def normal_pred_y_func(made_points: int):
    return made_points


def normal_strat_y_func(predicted_points: int, made_points: int):
    return -1 * np.absolute(predicted_points - made_points)


def prediction_vanilla_ffn():
    net = keras.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=(37,)),
        keras.layers.Dense(30, activation='relu'),
        keras.layers.Dense(1)
    ])
    net.compile(optimizer='rmsprop', loss='mse')
    return net


def strategy_vanilla_ffn(size_of_one_strat_net_input: int):
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


def prediction_multi_resnet():
    dense_output_size = 100
    net_input = Input(shape=(37,))
    layer_1 = Dense(dense_output_size, activation='relu')(net_input)
    layer_2 = Dense(dense_output_size, activation='relu')(layer_1)
    layer_3 = Dense(dense_output_size, activation='relu')(layer_2)
    res_sum = keras.layers.add([layer_1, layer_3])
    final_tensor = Dense(79)(res_sum)
    model = Model(inputs=net_input, outputs=final_tensor)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def strategy_resnet(use_batch_norm: bool, size_of_one_strat_net_input: int):
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


def strategy_rnn_resnet(use_batch_norm: bool):
    dense_output_size = 270
    rnn_output_size = 32
    rnn_input = Input(shape=(None, 9))
    rnn_output = LSTM(rnn_output_size)(rnn_input)
    aux_input = Input(shape=(87,))
    concat = keras.layers.concatenate([rnn_output, aux_input])
    net = Dense(dense_output_size)(concat)
    if use_batch_norm:
        net = BatchNormalization()(net)
    net = Activation('relu')(net)
    for _ in range(3):
        net = resnet_block(net, dense_output_size, use_batch_norm)
    final_tensor = Dense(1)(net)
    model = Model(inputs=[rnn_input, aux_input], outputs=final_tensor)
    model.compile(optimizer='rmsprop', loss='mse')
    return model