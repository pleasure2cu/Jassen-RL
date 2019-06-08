import keras
import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Dense, Activation, BatchNormalization

from helper_functions import resnet_block


def normal_pred_y_func(made_points: int):
    return made_points


def normal_strat_y_func(predicted_points: int, made_points: int) -> int:
    assert made_points is not None and 0 <= made_points <= 257
    return -1 * np.absolute(predicted_points - made_points)


def aggressive_strat_y_func(predicted_points: int, made_points: int) -> int:
    output = normal_strat_y_func(predicted_points, made_points)
    if made_points + 5 < predicted_points:
        output -= 8
    elif made_points < predicted_points:
        output -= 2
    return output


def defensive_strat_y_func(predicted_points: int, made_points: int) -> int:
    output = normal_strat_y_func(predicted_points, made_points)
    if made_points - 5 > predicted_points:
        output -= 8
    elif made_points > predicted_points:
        output -= 2
    return output


def very_aggressive_strat_y_func(predicted_points: int, made_points: int) -> int:
    output = normal_strat_y_func(predicted_points, made_points)
    output += made_points
    return output


def very_defensive_strat_y_func(predicted_points: int, made_points: int) -> int:
    output = normal_strat_y_func(predicted_points, made_points)
    output -= made_points
    return output


def prediction_resnet():
    dense_output_size = 50
    net_input = Input(shape=(37,))
    layer_1 = Dense(dense_output_size, activation='relu')(net_input)
    layer_3 = Dense(dense_output_size, activation='relu')(layer_1)
    res_sum = keras.layers.add([layer_1, layer_3])
    final_tensor = Dense(1)(res_sum)
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


def _deep_lstm():
    lstm_size = 270 // 5
    dense_size = 270 // 5
    inp = Input(shape=(None, 9))
    dense1 = Dense(dense_size, activation='relu')(inp)
    lstm1 = LSTM(lstm_size, return_sequences=True)(dense1)
    dense2 = Dense(dense_size, activation='relu')(lstm1)
    lstm2 = LSTM(lstm_size, return_sequences=True)(dense2)
    return inp, lstm2


def _deep_lstm2():
    lstm_size = 100
    inp = Input(shape=(None, 9))
    lstm1 = LSTM(lstm_size, return_sequences=True)(inp)
    lstm2 = LSTM(lstm_size, return_sequences=True)(lstm1)
    lstm3 = LSTM(lstm_size)(lstm2)
    return inp, lstm3


def strategy_deep_lstm_resnet():
    dense_size = 200
    lstm_in, lstm_out = _deep_lstm2()
    aux_inp = Input(shape=(47,))
    concat = keras.layers.concatenate([lstm_out, aux_inp])
    tmp = Dense(dense_size, activation='relu')(concat)
    tmp = Dense(dense_size, activation='relu')(tmp)
    out = Dense(1)(tmp)
    model = Model(inputs=[lstm_in, aux_inp], outputs=out)
    model.compile(optimizer='rmsprop', loss='mse')
    return model
