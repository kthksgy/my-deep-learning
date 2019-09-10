# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
import tensorflow as tf
from tensorflow import keras


def attention(shape: tuple, classes: int, depth: int) -> keras.Model:
    # Input
    inputs = keras.layers.Input(shape=shape)

    query = keras.layers.Dense(depth, use_bias=False, name='query_dense')(inputs)
    key = keras.layers.Dense(depth, use_bias=False, name='key_dense')(inputs)
    value = keras.layers.Dense(depth, use_bias=False, name='value_dense')(inputs)

    # Scaled Dot-Production
    query *= tf.sqrt(depth)
    logit = tf.matmul(query, key, transpose_b=True)

    attention_weight = keras.layers.Softmax(name='attention_weight')(logit)

    attention_output = tf.matmul(value, attention_weight)

    output = keras.layers.Dense(depth, use_bias=False, name='output')(attention_output)
    # TODO: 続きを作る
