# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def model_vgg16_keras(shape: tuple, classes: int) -> keras.Model:
    """
    Keras Applicationsに用意されているVGG16を読み込む。

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    Karen Simonyan, Andrew Zisserman
    https://arxiv.org/abs/1409.1556

    Args:
        shape tuple:
            入力の形状を指定する。
        classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            VGG16を返す。
    """
    return keras.applications.vgg16.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=shape,
        pooling=None,
        classes=classes
    )


def vgg16_1d(shape: tuple, classes: int) -> keras.Model:
    """
    一次元版のVGG16を読み込む。

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    Karen Simonyan, Andrew Zisserman
    https://arxiv.org/abs/1409.1556

    Args:
        shape tuple:
            入力の形状を指定する。
        classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            VGG16を返す。
    """
    return keras.Sequential([
        keras.layers.InputLayer(shape),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(256, 1, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 1, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(512, 1, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(classes, activation='softmax')
    ], name='vgg16_1d')
