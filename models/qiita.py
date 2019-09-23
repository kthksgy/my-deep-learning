# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def model_qiita(
        batch_shape: tuple, num_classes: int,
        units=100,
        vocabularies=258,
        lstm_layers=2,
        conv_layers=2, enable_bn=True):
    """\
    Qiitaに掲載されている圧縮バイナリ認識のモデルを読み込む。
    元のモデルはChainerで実装されているので、それと等価なKerasのモデルになっている。

    参考文献:
    https://qiita.com/Hi-king/items/7c61f54986d9a940208f
    https://github.com/Hi-king/compressed_image_recognition/blob/master/compressed_image_recoginition/models.py
    Args:
        batch_shape tuple:
            入力バッチの形状。
        num_classes int:
            分類するクラス数。
        units int:
            RNNのユニット数。
        vocabularies int:
            入力語彙数。
        lstm_layers int:
            LSTMのレイヤー数。
        conv_layers int:
            一次元畳み込みのレイヤー数。
        enable_bn bool:
            畳み込みの後にバッチ正規化を行うか。
    Returns:
        keras.Model:
            Qiitaのモデルを返す。
    """
    INPUT_LENGTH = batch_shape[-1]
    inputs = keras.layers.Input(batch_shape=batch_shape)
    x = keras.layers.Embedding(
        vocabularies, units,
        input_length=INPUT_LENGTH,
        mask_zero=True,
        name='embedding')(inputs)

    for i in range(conv_layers):
        x = keras.layers.Conv1D(units, 3, name='convolution_%d' % i)(x)
        if enable_bn:
            x = keras.layers.BatchNormalization(
                name='batch_normalization_%d' % i)(x)
    cells = [
        keras.layers.LSTMCell(units)
        for _ in range(lstm_layers)
    ]
    x = keras.layers.RNN(
        keras.layers.StackedRNNCells(cells),
        return_sequences=False, name='lstm_%d_layers_stacked' % lstm_layers)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='qiita')
