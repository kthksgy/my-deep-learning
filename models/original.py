# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def model_original(
        batch_shape: tuple, num_classes: int,
        cell='lstm',
        num_vocabularies=258,
        units=64,
        num_rnn_layers=1,
        dropout_rate=0.1
        ) -> keras.Model:
    """\
    num_rnn_layers層unitsユニットのRNNからなるモデルを読み込む。
    Args:
        batch_shape tuple:
            入力バッチの形状。
        num_classes int:
            分類するクラス数。
        units int:
            RNNのユニット数。
        num_rnn_layers int:
            RNNのレイヤー数。
        dropout_rate int:
            RNNの各層の入力時のドロップアウト率。
    Returns:
        keras.Model:
            オリジナルのRNNモデルを返す。
    """
    INPUT_LENGTH = batch_shape[-1]
    inputs = keras.layers.Input(batch_shape=batch_shape)
    x = keras.layers.Embedding(
        num_vocabularies, units,
        input_length=INPUT_LENGTH,
        # CuDNN版RNNは無効値0をサポートしてない
        mask_zero=False if cell.lower().startswith('cudnn') else True,
        name='embedding')(inputs)
    # KerasのGRU
    if cell.lower() == 'gru':
        cells = [
            keras.layers.GRUCell(
                units, dropout=dropout_rate, reset_after=True)
            for i in range(num_rnn_layers - 1)
        ]
        x = keras.layers.RNN(
            keras.layers.StackedRNNCells(cells),
            return_sequences=False, name='gru_%d_stacked' % num_rnn_layers)(x)
    # KerasのLSTM
    elif cell.lower() == 'lstm':
        cells = [
            keras.layers.LSTMCell(units, dropout=dropout_rate)
            for _ in range(num_rnn_layers - 1)
        ]
        x = keras.layers.RNN(
            keras.layers.StackedRNNCells(cells),
            return_sequences=False, name='lstm_%d_stacked' % num_rnn_layers)(x)
    # # CuDNN版のGRU
    # elif cell.lower() == 'cudnngru':
    #     for _ in range(num_rnn_layers - 1):
    #         # ドロップアウト率が指定されていたら層を追加
    #         if dropout_rate > 0:
    #             x = keras.layers.Dropout(dropout_rate)(x)
    #         x = keras.layers.CuDNNGRU(units, return_sequences=True)(x)
    # # CuDNN版のLSTM
    # elif cell.lower() == 'cudnnlstm':
    #     for _ in range(num_rnn_layers - 1):
    #         # ドロップアウト率が指定されていたら層を追加
    #         if dropout_rate > 0:
    #             x = keras.layers.Dropout(dropout_rate)(x)
    #         x = keras.layers.CuDNNLSTM(units, return_sequences=True)(x)
    # 通常のRNN
    else:
        cells = [
            keras.layers.SimpleRNNCell(
                units, dropout=dropout_rate,
                return_sequences=True)
            for _ in range(num_rnn_layers - 1)
        ]
        x = keras.layers.RNN(
            keras.layers.StackedRNNCells(cells),
            return_sequences=False, name='rnn_%d_stacked' % num_rnn_layers)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="original")


def model_original_dense(input_shape: tuple, num_classes: int):
    return keras.models.Sequential([
        keras.layers.Conv2D(16, 1, 1, 'same', input_shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
