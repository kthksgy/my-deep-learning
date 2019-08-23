# -*- coding: utf-8 -*-
"""
TensorFlow用のデータセットを読み込む時の補助を行う。
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


class DatasetLoader:
    """
    データセットの読み込みを補助するクラス
    """
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir str:
                データセットを読み込むパスを指定する。
        """
        self.__load_kwargs = {
            'split': None,
            'data_dir': data_dir,
            'download': True,
            'as_supervised': True,
            'decoders': None,
            'with_info': True,
            'builder_kwargs': None,
            'download_and_prepare_kwargs': None,
            'as_dataset_kwargs': None,
            'try_gcs': True
        }
        self.__call__ = self.load

    def load(self, name: str, batch_size: int, in_memory=True) -> \
            (dict, tfds.core.DatasetInfo):
        """
        tfds.load()のラッパー
        https://www.tensorflow.org/datasets/api_docs/python/tfds/load

        Args:
            name str:
                データセット名を指定する。
            batch_size int:
                1バッチに含まれるデータ数を指定する。
            in_memory bool:
                読み込んだデータをメモリ上に格納する。
                データの読み込み時間を削減出来る。

        Returns:
            ({str:tf.data.Dataset}, tfds.core.DatasetInfo):
                種類(train, validation, test)をキーとしたデータセットの辞書と情報を返す。
        """
        datasets, info = tfds.load(
            name,
            batch_size=batch_size,
            in_memory=in_memory,
            **self.__load_kwargs
        )
        return datasets, info
