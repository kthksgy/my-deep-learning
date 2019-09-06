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

    def dct1(self, batch_shape, axis=-1):
        # TODO: TFのdctのaxisが機能するようになったらそちらに差し替える
        transposed_shape = list(range(len(batch_shape) - 1))
        if 0 <= axis and axis < len(transposed_shape):
            transposed_shape[-1], transposed_shape[axis] =\
                transposed_shape[axis], transposed_shape[-1]

        tmp = """def map_func(data, label):
            transposed = tf.transpose(data, %(transposed_shape)s)
            dcted = tf.signal.dct(transposed, norm='ortho')
            untransposed = tf.transpose(dcted, %(transposed_shape)s)
            reshaped = tf.reshape(
                dcted,
                (%(batch_shape_0)s, %(batch_shape_axis)s, -1)
            )
            return reshaped, label""" % {
                'transposed_shape': transposed_shape,
                'batch_shape_0': batch_shape[0],
                'batch_shape_axis': batch_shape[axis]
            }
        exec(tmp, globals())
        return map_func

    def encode_jpeg(self, quality=50, progressive=False, optimize_size=False, chroma_downsampling=True, skip_header=0):
        tmp = """def map_func(image, label):
            image = tf.image.encode_jpeg(
                image,
                quality=%(quality)d,
                progressive=%(progressive)s,
                optimize_size=%(optimize_size)s,
                chroma_downsampling=%(chroma_downsampling)s
            )
            image = tf.io.decode_raw(image, tf.uint8)
        """ % {
            'quality': quality,
            'progressive': progressive,
            'optimize_size': optimize_size,
            'chroma_downsampling': chroma_downsampling
        }
        if skip_header > 0:
            tmp += """
            marker_pos = tf.where(tf.equal(image, 255))
            image = image[marker_pos[tf.where(tf.equal(tf.gather(image, marker_pos + 1), 218))[0][0]][0] + 8 + %d * 2:-3]
        """ % (skip_header)
        tmp += """
            return tf.cast(image, tf.int16), label
        """
        exec(tmp, globals())
        return map_func

    def encode_png(self, compression=50, skip_header=0):
        tmp = """def map_func(image, label):
            image = tf.image.encode_jpeg(
                image,
                compression=%(compression)d
            )
            image = tf.io.decode_raw(image, tf.uint8)
        """ % {'compression': compression}
        if skip_header > 0:
            # ヘッダー
            # PNGファイルシグネチャ(8バイト) + IHDR(25バイト) = 33バイト
            # フッター
            # IEND(12バイト) = 12バイト
            # 参考文献: PNG ファイルフォーマット(https://www.setsuki.com/hsp/ext/png.htm)
            tmp += """
            image = image[33:-12]
        """ % (skip_header)
        tmp += """
            return tf.cast(image, tf.int16), label
        """
        exec(tmp, globals())
        return map_func
