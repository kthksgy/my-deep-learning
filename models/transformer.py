# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
import tensorflow as tf
from tensorflow import keras
from .layers import LayerNormalization, PositionalEncoding


def model_transformer_encoder(
    batch_shape: tuple,
    num_classes: int,
    num_vocabularies=258,
    embedding_dim=64,
    hidden_dim=64,
    num_heads=2,
    dropout_rate=0.1,
    dtype='int16',
    hopping=4,
    pad_id=0,
    num_conv_layers=0,
    conv_kernel_size=3
) -> keras.Model:
    # 必要な数値を変数に保持
    BATCH_SIZE = batch_shape[0]
    INPUT_LENGTH = batch_shape[-1]
    # Input
    inputs = keras.layers.Input(
        batch_shape=batch_shape,
        dtype=dtype,
        name='input')

    query = keras.layers.Embedding(
        num_vocabularies, embedding_dim,
        input_length=INPUT_LENGTH,
        name='embedding')(inputs)

    for i in range(num_conv_layers):
        query = keras.layers.Conv1D(
            hidden_dim, conv_kernel_size, name='conv_%d' % i)(query)

    query = PositionalEncoding(name='positional_encoding')(query)

    # shape broadcasting で [batch_size, head_num, (m|q)_length, m_length] になる
    attention_mask = keras.backend.cast(
        keras.backend.reshape(
            keras.backend.equal(
                inputs, pad_id),
            (BATCH_SIZE, 1, 1, INPUT_LENGTH)),
        keras.backend.floatx())
    attention_mask *= attention_mask.dtype.min

    # Multi-Headにするための分割/結合用レイヤー
    multi_head_transpose = keras.layers.Permute(
        (2, 1, 3),
        name='multi_head_transpose')
    split_reshape = keras.layers.Reshape(
        (INPUT_LENGTH, num_heads, hidden_dim // num_heads),
        name='split_head_reshape')
    combine_reshape = keras.layers.Reshape(
        (INPUT_LENGTH, hidden_dim),
        name='combine_head_reshape')

    for i in range(hopping):
        # Attention PreProcessing
        base = query
        query = LayerNormalization(name='attention_pre_norm_h%d' % i)(query)

        # Self Multi-Head Attention
        query = keras.layers.Dense(
            hidden_dim, use_bias=False, name='query_dense_h%d' % i)(query)
        key = keras.layers.Dense(
            hidden_dim, use_bias=False, name='key_dense_h%d' % i)(query)
        value = keras.layers.Dense(
            hidden_dim, use_bias=False, name='value_dense_h%d' % i)(query)

        query = split_reshape(query)
        query = multi_head_transpose(query)
        key = split_reshape(key)
        key = multi_head_transpose(key)
        value = split_reshape(value)
        value = multi_head_transpose(value)

        # Scaled Dot-Production
        _scale = keras.backend.constant(
            hidden_dim // num_heads, dtype=keras.backend.floatx())
        _scale = keras.backend.sqrt(_scale)
        query /= _scale
        query = tf.matmul(query, key, transpose_b=True)
        query += attention_mask

        query = keras.layers.Softmax(name='attention_weight_h%d' % i)(query)
        query = keras.layers.Dropout(
            dropout_rate, name='attention_dropout_h%d' % i)(query)

        query = tf.matmul(query, value)

        query = multi_head_transpose(query)
        query = combine_reshape(query)

        query = keras.layers.Dense(
            hidden_dim, use_bias=False, name='attention_output_h%d' % i)(query)

        # Attention PostProcessing
        query = keras.layers.Dropout(
            dropout_rate, name='attention_post_do_h%d' % i)(query)
        query = keras.layers.Add(
            name='attention_rescon_h%d' % i)([base, query])

        # FFN PreProcessing
        base = query
        query = LayerNormalization(name='ffn_pre_norm_h%d' % i)(query)

        # Feed Forward Network
        query = keras.layers.Dense(
            hidden_dim * 4, use_bias=True, activation='relu',
            name='ffn_filter_h%d' % i)(query)
        query = keras.layers.Dropout(
            dropout_rate, name='ffn_dropout_h%d' % i)(query)
        query = keras.layers.Dense(
            hidden_dim, use_bias=True, activation='relu',
            name='ffn_output_h%d' % i)(query)

        # FFN PostProcessing
        query = keras.layers.Dropout(
            dropout_rate, name='ffn_post_do_h%d' % i)(query)
        query = keras.layers.Add(name='ffn_rescon_h%d' % i)([base, query])

    cls_vector = keras.backend.permute_dimensions(query, (1, 0, 2))
    cls_vector = keras.backend.gather(cls_vector, [0])
    cls_vector = keras.backend.squeeze(cls_vector, 0)
    outputs = keras.layers.Dense(
        num_classes, activation='softmax',
        name='final_output_dense')(cls_vector)
    return keras.Model(
        inputs=inputs, outputs=outputs,
        name='transformer_encoder')
