# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def model_xception_keras(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    Keras Applicationsに用意されているXceptionを読み込む。

    Xception: Deep Learning with Depthwise Separable Convolutions
    François Chollet
    https://arxiv.org/abs/1610.02357

    Args:
        input_shape tuple:
            入力の形状を指定する。
        num_classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            Xceptionを返す。
    """
    return keras.applications.xception.Xception(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=num_classes
    )


def model_xception(input_shape: tuple, num_classes: int, entry=True, middle=True) -> keras.Model:
    """
    Xceptionを読み込む。

    Xception: Deep Learning with Depthwise Separable Convolutions
    François Chollet
    https://arxiv.org/abs/1610.02357

    Args:
        input_shape tuple:
            入力の形状を指定する。
        num_classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            Xceptionを返す。
    """
    # Input
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs

    if entry:
        # Entry Flow
        x = keras.layers.Conv2D(32, 3, 2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        res = keras.layers.Conv2D(128, 1, 2, padding='valid')(x)
        res = keras.layers.BatchNormalization()(res)

        x = keras.layers.SeparableConv2D(128, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.ReLU()(x)
        x = keras.layers.SeparableConv2D(128, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

        x = keras.layers.Add()([x, res])

        res = keras.layers.Conv2D(256, 1, 2, padding='valid')(x)
        res = keras.layers.BatchNormalization()(res)

        x = keras.layers.ReLU()(x)
        x = keras.layers.SeparableConv2D(256, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.ReLU()(x)
        x = keras.layers.SeparableConv2D(256, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

        x = keras.layers.Add()([x, res])

        res = keras.layers.Conv2D(728, 1, 2, padding='valid')(x)
        res = keras.layers.BatchNormalization()(res)

        x = keras.layers.ReLU()(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.ReLU()(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

        x = keras.layers.Add()([x, res])
    else:
        x = keras.layers.Conv2D(728, 1, 1, 'same')(x)

    if middle:
        # Middle Flow
        for i in range(8):
            res = x
            x = keras.layers.SeparableConv2D(728, 3, padding='same', name='middle_sepconv1_%d' % (i + 1))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SeparableConv2D(728, 3, padding='same', name='middle_sepconv2_%d' % (i + 1))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SeparableConv2D(728, 3, padding='same', name='middle_sepconv3_%d' % (i + 1))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Add()([x, res])

    # Exit Flow
    res = keras.layers.Conv2D(1024, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    x = keras.layers.SeparableConv2D(1536, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(2048, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def model_xception_1d(shape: tuple, classes: int) -> keras.Model:
    """
    一次元版のXceptionを読み込む。

    Xception: Deep Learning with Depthwise Separable Convolutions
    François Chollet
    https://arxiv.org/abs/1610.02357

    Args:
        shape tuple:
            入力の形状を指定する。
        classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            Xceptionを返す。
    """
    # Input
    inputs = keras.layers.Input(shape=shape)
    x = inputs

    # Entry Flow
    x = keras.layers.Conv1D(32, 3, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(64, 3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    res = keras.layers.Conv1D(128, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.SeparableConv1D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool1D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    res = keras.layers.Conv1D(256, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(256, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool1D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    res = keras.layers.Conv1D(728, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool1D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    # Middle Flow
    for i in range(8):
        res = x
        x = keras.layers.SeparableConv1D(728, 3, padding='same', name='middle_sepconv1_%d' % (i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv1D(728, 3, padding='same', name='middle_sepconv2_%d' % (i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv1D(728, 3, padding='same', name='middle_sepconv3_%d' % (i + 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, res])

    # Exit Flow
    res = keras.layers.Conv1D(1024, 1, 2, padding='valid')(x)
    res = keras.layers.BatchNormalization()(res)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ReLU()(x)
    x = keras.layers.SeparableConv1D(1024, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool1D(3, 2, padding='same')(x)

    x = keras.layers.Add()([x, res])

    x = keras.layers.SeparableConv1D(1536, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv1D(2048, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)

    outputs = keras.layers.Dense(classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="xception_1d")
