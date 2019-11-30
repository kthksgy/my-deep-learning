# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def model_dwt(input_shape: tuple, num_classes: int) -> keras.Model:
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
    
    x = keras.layers.Conv2D(64, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, 5, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(128, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(512, 1, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(512, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(768, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(1024, 1, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, 'softmax')(x)

    outputs = x

    return keras.Model(inputs=inputs, outputs=outputs, name='Orig1')