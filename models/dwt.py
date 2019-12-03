# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
import tensorflow as tf
from tensorflow import keras


def model_dwt(input_shape: tuple, num_classes: int, batch_size=None) -> keras.Model:
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

    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
    x = inputs
    
    x = keras.layers.Conv2D(32, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(48, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, 5, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(64, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(96, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, 5, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(128, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(160, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(192, 5, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(192, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(224, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(256, 5, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, 'softmax')(x)

    outputs = x

    return keras.Model(inputs=inputs, outputs=outputs, name='Orig1')

def model_dwt_lateconcat(input_shape: tuple, num_classes: int, batch_size=None) -> keras.Model:
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

    inputs = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = inputs

    bs = x.shape[0]
    h = x.shape[-3]
    w = x.shape[-2]
    c = x.shape[-1]
    xs = []
    imax = 3
    for i in range(1, imax):
        xs.append(
            tf.concat([
                tf.slice(x, [0,0,w//pow(2,i),0], [bs,h//pow(2,i),w//pow(2,i),c]),
                tf.slice(x, [0,h//pow(2,i),0,0], [bs,h//pow(2,i),w//pow(2,i),c]),
                tf.slice(x, [0,h//pow(2,i),w//pow(2,i),0], [bs,h//pow(2,i),w//pow(2,i),c]),
            ], -1)
        )
    xs.append(
        tf.concat([
            tf.slice(x, [0,0,0,0], [bs,h//pow(2,imax),w//pow(2,imax),c]),
            tf.slice(x, [0,0,w//pow(2,imax),0], [bs,h//pow(2,imax),w//pow(2,imax),c]),
            tf.slice(x, [0,h//pow(2,imax),0,0], [bs,h//pow(2,imax),w//pow(2,imax),c]),
            tf.slice(x, [0,h//pow(2,imax),w//pow(2,imax),0], [bs,h//pow(2,imax),w//pow(2,imax),c]),
        ], -1)
    )

    x = keras.layers.Conv2D(64, 1, 1, 'same', activation='relu')(xs[0])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(96, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, 5, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = tf.concat([x, xs[1]], -1)

    x = keras.layers.Conv2D(128, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(160, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(192, 5, 2, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = tf.concat([x, xs[2]], -1)

    x = keras.layers.Conv2D(192, 1, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(224, 3, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(256, 5, 1, 'same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, 'softmax')(x)

    outputs = x

    return keras.Model(inputs=inputs, outputs=outputs, name='Orig1 Late Concat')