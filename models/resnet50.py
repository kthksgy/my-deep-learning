# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def model_resnet50_keras(
    input_shape: tuple, classes: int,
    include_top=True, weights='imagenet',) -> keras.Model:
    """
    Keras Applicationsに用意されているResNet50を読み込む。

    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    https://arxiv.org/abs/1512.03385

    Args:
        input_shape tuple:
            入力の形状を指定する。
        num_classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            ResNet50を返す。
    """
    return keras.applications.resnet50.ResNet50(
        include_top=include_top,
        weights=weights,
        input_tensor=None,
        input_shape=(224, 224, 3) if weights == 'imagenet' else input_shape,
        pooling=None,
        classes=classes if include_top and not weights else 1000
    )


def model_resnet50(input_shape: tuple, classes: int) -> keras.Model:
    """
    ResNet50を読み込む。

    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    https://arxiv.org/abs/1512.03385

    Args:
        input_shape tuple:
            入力の形状を指定する。
        num_classes int:
            分類するクラス数を指定する。

    Returns:
        keras.Model:
            ResNet50を返す。
    """
    # Input
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    s = max(input_shape[0], input_shape[1])
    if s >= 224:
        x = keras.layers.Conv2D(64, 7, 2, name='conv1')(x)
    
    if s >= 112:
        pass 

    # Entry Flow
    x = keras.layers.Conv2D(32, 3, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)