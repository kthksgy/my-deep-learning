# -*- coding: utf-8 -*-
"""
TensorFlow Kerasのモデルを読み込む時の補助を行う。
"""
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras

__author__ = 'sugaok <https://github.com/sugaok>'
__status__ = 'production'
__version__ = '0.0.1'
__date__ = '21 August 2019'


class ModelLoader:
    """
    モデルの読み込みを補助するクラス
    """
    def __init__(self, save_dir=r"./"):
        """
        Args:
            save_dir str:
                モデルを保存するパスを指定する。
        """
        self.MODEL_FUNC_PREFIX = "model_"
        if(not save_dir or len(save_dir) == 0):
            self.save_dir = r"./"
        elif(save_dir[-1] != r"/"):
            self.save_dir = save_dir + r"/"
        else:
            self.save_dir = save_dir
        self.__call__ = self.load

    def load(self, shape: tuple, classes: int, name=None) -> \
            keras.Model:
        """
        モデルを読み込む。
        モデル名を指定しなければ、モデル名一覧を表示しユーザーに入力を促す。

        Args:
            shape tuple:
                入力の形状を指定する。
            classes int:
                分類するクラス数を指定する。
            name str:
                モデル名を指定する。

        Returns:
            keras.Model:
                Kerasのモデルを返す。
        """
        if not name:
            model_names = [
                s[len(self.MODEL_FUNC_PREFIX):]
                for s in ModelLoader.__dict__
                if s.startswith(self.MODEL_FUNC_PREFIX)
            ]
            print("モデル一覧: {}".format(", ".join(model_names)))
            name = input("モデル名を入力してください -> ")
        return eval("self.{}{}({}, {})".format(
            self.MODEL_FUNC_PREFIX,
            name,
            str(shape),
            str(classes))
        )

    def model_xception_keras(self, shape: tuple, classes: int) -> keras.Model:
        """
        Keras Applicationsに用意されているXceptionを読み込む。

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
        return keras.applications.xception.Xception(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=shape,
            pooling=None,
            classes=classes
        )

    def model_xception(self, shape: tuple, classes: int) -> keras.Model:
        """
        Xceptionを読み込む。

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

        outputs = keras.layers.Dense(classes, activation='softmax')(x)

        return keras.Model(inputs=inputs, outputs=outputs)

    def model_xception_1d(self, shape: tuple, classes: int) -> keras.Model:
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

    def model_vgg16_keras(self, shape: tuple, classes: int) -> keras.Model:
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

    def model_vgg16_1d(self, shape: tuple, classes: int) -> keras.Model:
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

    def save(self, model: keras.Model):
        model.save(self.save_dir + "{}.h5".format(model.name))
