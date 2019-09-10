# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def sample(input_shape: tuple, num_classes: int, units=1024) -> keras.Model:
    """\
    サンプルモデルを読み込む。

    Args:
        input_shape tuple:
            入力の形状を指定する。
        num_classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            VGG16を返す。
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units, activation='relu')(x)
    x = keras.layers.Dense(units * 2, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="sample")
