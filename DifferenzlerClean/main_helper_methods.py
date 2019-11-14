import keras
import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Dense, Activation, BatchNormalization, SimpleRNN, CuDNNLSTM
from keras import backend as K


def resnet_block(input_tensor, layer_size: int, use_batch_norm: bool):
    """
    implements one resnet block. Meaning:
        Input_tensor ---> Dense -> BN -> ReLU -> Dense -> BN ---> ReLU
                      |                                       |
                       ---------------------------------------
    the resulting tensor of the last ReLU will be the return
    :param input_tensor: as name says
    :param layer_size: size of the output of the input_tensor
    :param use_batch_norm: bool-flag
    :return: tensor from the last ReLU
    """
    block = keras.layers.Dense(layer_size)(input_tensor)
    if use_batch_norm:
        block = keras.layers.BatchNormalization()(block)
    block = keras.layers.Activation('relu')(block)
    block = keras.layers.Dense(layer_size)(block)
    if use_batch_norm:
        block = keras.layers.BatchNormalization()(block)
    block = keras.layers.add([block, input_tensor])
    return keras.layers.Activation('relu')(block)


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


def prediction_resnet(loss='mse'):
    dense_output_size = 50
    net_input = Input(shape=(37,))
    layer_1 = Dense(dense_output_size, activation='relu')(net_input)
    layer_2 = Dense(dense_output_size, activation='relu')(layer_1)
    res_sum = keras.layers.add([layer_1, layer_2])
    final_tensor = Dense(1)(res_sum)
    model = Model(inputs=net_input, outputs=final_tensor)
    model.compile(optimizer='rmsprop', loss=loss)
    return model


def prediction_l1_resnet():
    return prediction_resnet(keras.losses.mean_absolute_error)


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


def _deep_lstm2(lstm_size=100):
    inp = Input(shape=(None, 9))
    lstm_f = LSTM if len(K.tensorflow_backend._get_available_gpus()) == 0 else CuDNNLSTM
    lstm1 = lstm_f(lstm_size, return_sequences=True)(inp)
    lstm2 = lstm_f(lstm_size, return_sequences=True)(lstm1)
    lstm3 = lstm_f(lstm_size)(lstm2)
    return inp, lstm3


def _deep_simple_rnn(size=100):
    inp = Input(shape=(None, 9))
    rnn1 = SimpleRNN(size, return_sequences=True)(inp)
    rnn2 = SimpleRNN(size, return_sequences=True)(rnn1)
    rnn3 = SimpleRNN(size)(rnn2)
    return inp, rnn3


def strategy_deep_lstm_resnet(lstm_size=100, dense_size=200, loss='mse'):
    lstm_in, lstm_out = _deep_lstm2(lstm_size)
    aux_inp = Input(shape=(47,))
    concat = keras.layers.concatenate([lstm_out, aux_inp])
    tmp = Dense(dense_size, activation='relu')(concat)
    tmp = Dense(dense_size, activation='relu')(tmp)
    out = Dense(1)(tmp)
    model = Model(inputs=[lstm_in, aux_inp], outputs=out)
    model.compile(optimizer='rmsprop', loss=loss)
    return model


def strategy_deep_simple_rnn_resnet(rnn_size=100, dense_size=200):
    rnn_in, rnn_out = _deep_simple_rnn(rnn_size)
    aux_inp = Input(shape=(47,))
    concat = keras.layers.concatenate([rnn_out, aux_inp])
    tmp = Dense(dense_size, activation='relu')(concat)
    tmp = Dense(dense_size, activation='relu')(tmp)
    out = Dense(1)(tmp)
    model = Model(inputs=[rnn_in, aux_inp], outputs=out)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def normal_strategy_network():
    return strategy_deep_lstm_resnet()


def small_strategy_network():
    return strategy_deep_lstm_resnet(70, 140)


def tiny_strategy_network():
    dense_output_size = 140
    rnn_output_size = 70
    rnn_input = Input(shape=(None, 9))
    rnn_output = LSTM(rnn_output_size)(rnn_input) if len(K.tensorflow_backend._get_available_gpus()) == 0 \
        else CuDNNLSTM(rnn_output_size)(rnn_input)
    aux_input = Input(shape=(47,))
    concat = keras.layers.concatenate([rnn_output, aux_input])
    net = Dense(dense_output_size, activation='relu')(concat)
    net = Dense(dense_output_size, activation='relu')(net)
    out = Dense(1)(net)
    model = Model(inputs=[rnn_input, aux_input], outputs=out)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def small_rnn_strategy_network():
    return strategy_deep_simple_rnn_resnet(70, 140)


def small_l1_strategy_network():
    return strategy_deep_lstm_resnet(70, 140, keras.losses.mean_absolute_error)


def hand_crafted_features_rnn_network(use_batch_norm=True) -> keras.Model:
    rnn_output_size = 70
    rnn_in, rnn_out = _deep_simple_rnn(rnn_output_size)
    # inputs for aux
    aux_input = Input(
        (140,),
        name="36_hand_cards_8_relative_table_1_current_diff_36_gone_cards_36_bocks_16_could_follow_1_points_on_table_4_made_points_2_action"
    )
    # putting together the rest of the network
    feed_forward_input = keras.layers.concatenate([
        rnn_out, aux_input
    ])
    scale_down = Dense(130, activation='relu')(feed_forward_input)
    first_block = resnet_block(scale_down, 130, use_batch_norm)
    scnd_block = resnet_block(first_block, 130, use_batch_norm)
    out = Dense(1)(scnd_block)
    model = Model(inputs=[rnn_in, aux_input], outputs=out)
    model.compile(optimizer='rmsprop', loss='mse')
    return model
