# -*- coding: UTF-8 -*-
from __future__ import (
    absolute_import, division,
    print_function, unicode_literals)

import argparse
import glob
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow import keras

import tfds_e


def get_model_names():
    model_names = []
    for filename in glob.glob(
        os.path.normcase(
            os.path.join(
                os.path.dirname(__file__), 'models/*.py'))):
        script = """\
from models import %(filename)s
model_names.extend([
    name[%(prefix_len)d:] for name in dir(%(filename)s)
    if name.startswith('model_')])
        """ % {
            'prefix_len': len('model_'),
            'filename': os.path.basename(filename)[:-len('.py')]}
        exec(script)
    return model_names


def main():
    # コマンドライン引数で簡単なプログラムの変更は出来るように
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help='モデル名を指定します')
    parser.add_argument(
        '-d', '--dataset',
        help='データセット名を指定します')
    parser.add_argument(
        '-b', '--batch', type=int,
        help='バッチサイズを指定します')
    parser.add_argument(
        '-lm', '--listmodels',
        help='モデル名の一覧を表示します。',
        action="store_true")
    parser.add_argument(
        '-mk', '--modelkwargs', default='{}',
        help='モデルに渡す追加パラメータを指定します。')
    args = parser.parse_args()

    if args.listmodels:
        print('モデル名: ', get_model_names())
        return

    # バージョン表示
    print('Python Version: ', sys.version)
    print('TensorFlow Version: ', tf.__version__)
    print('Keras Version: ', keras.__version__)

    # TensorFlow 2ではない場合はEager Executionを有効にする
    if not tf.executing_eagerly():
        tf.enable_eager_execution()
    # 乱数種値を固定
    tf.random.set_random_seed(0)        # TensorFlow
    os.environ['PYTHONHASHSEED'] = '0'  # PYTHONのハッシュ関数
    np.random.seed(0)                   # numpy
    random.seed(0)                      # 標準のランダム
    # tf.debugging.set_log_device_placement(True)

    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch
    MODEL_NAME = args.model
    _tmp = MODEL_NAME.find('_')
    MODEL_FILENAME = MODEL_NAME if _tmp == -1 else MODEL_NAME[:_tmp]

    INPUT_LENGTH = 100

    # データセットの読み込み
    datasets, info = tfds_e.load(DATASET_NAME)

    # データセットへの前処理
    for key in datasets:
        # 前処理
        datasets[key] = datasets[key].map(
            tfds_e.map_encode_jpeg(quality=0, skip_header=True), 16)
        datasets[key] = datasets[key].map(
            lambda image, label: (tf.cast(image, tf.int16), label), 16)
        datasets[key] = datasets[key].map(tfds_e.map_add_cls(cls_id=256), 16)
        datasets[key] = datasets[key].map(tfds_e.map_shift_id(amount=1), 16)
        # シャッフルやデータオーギュメンテーション等は訓練用データのみに適用
        if key == 'train':
            datasets[key] = datasets[key].shuffle(BATCH_SIZE * 5)
        # ミニバッチ化
        datasets[key] = datasets[key].padded_batch(
            BATCH_SIZE,
            ((INPUT_LENGTH), ()),
            drop_remainder=True
        )
        # 事前読み込みのパラメータ―1で自動調整モード
        datasets[key] = datasets[key].prefetch(-1)

    # INPUT_SHAPE = (BATCH_SIZE,) + info.features['image'].shape
    INPUT_SHAPE = (BATCH_SIZE, INPUT_LENGTH)
    NUM_CLASSES = info.features['label'].num_classes

    MODEL_NAME_WITH_INPUT_SHAPE = 'model_%s_%s' % (
        MODEL_NAME, '_'.join(map(str, INPUT_SHAPE)))
    SAVED_MODEL_WEIGHTS_FILENAME = MODEL_NAME_WITH_INPUT_SHAPE + '.h5'

    # モデルの読み込み
    if os.path.exists(SAVED_MODEL_WEIGHTS_FILENAME):
        # 保存されたデータが有ればコンパイル無しで読み込む
        print('LAODODODODOOMOD')
        model = keras.models.load_model(
            SAVED_MODEL_WEIGHTS_FILENAME,
            compile=False)
    else:
        print('LDODOD')
        # 新たなモデルを読み込む

#         script = """\
# from models.%(filename)s import model_%(modelname)s as load_model
# model = model_%(modelname)s(%(input_shape)s, %(num_classes)d, **%(kwargs)s)\
#     """ % {
#             'filename': MODEL_FILENAME,
#             'modelname': MODEL_NAME,
#             'input_shape': INPUT_SHAPE,
#             'num_classes': NUM_CLASSES,
#             'kwargs': args.modelkwargs
#         }
        script = 'from models.%(filename)s\
            import model_%(modelname)s as load_model' % {
            'filename': MODEL_FILENAME,
            'modelname': MODEL_NAME
        }
        exec(script, globals())
        model = load_model(INPUT_SHAPE, NUM_CLASSES)
    if not model:
        print('モデルの読み込みに失敗しました。')
        return

    loss_object = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Nadam()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    model.compile(optimizer=optimizer, loss=loss_object)
    model.summary()
    try:
        keras.utils.plot_model(
            model, MODEL_NAME_WITH_INPUT_SHAPE + '.png', True, True, 'TB')
    except ImportError as e:
        print('graphvizとpydotが見つからないため、モデルの図の出力は出来ません。')

    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, predictions)

    @tf.function
    def test_step(image, label):
        predictions = model(image)
        loss = loss_object(label, predictions)

        test_loss(loss)
        test_accuracy(label, predictions)

    file_result = open('result.csv', mode='a')
    file_result.write('epochs,train_loss,train_acc,test_loss,test_acc\n')

    EPOCHS = 1000

    for epoch in range(EPOCHS):
        keras.backend.set_learning_phase(1)
        for image, label in datasets['train']:
            train_step(image, label)

        if('validation' in datasets):
            keras.backend.set_learning_phase(0)
            for test_image, test_label in datasets['validation']:
                test_step(test_image, test_label)

        if('test' in datasets):
            keras.backend.set_learning_phase(0)
            for test_image, test_label in datasets['test']:
                test_step(test_image, test_label)

        template = 'Epoch {:0%d}, Loss: {:.5f}, Accuracy: {:.4f}, Test Loss: {:.5f}, Test Accuracy: {:.4f}' % len(str(EPOCHS))  # noqa: E501
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result(),
                test_loss.result(),
                test_accuracy.result()
            )
        )
        result_line = '{:0%d},{:.5f},{:.4f},{:.5f},{:.4f}\n' % len(str(EPOCHS))
        file_result.write(
            result_line.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result(),
                test_loss.result(),
                test_accuracy.result()
            )
        )
    file_result.close()

    # モデルの保存
    keras.models.save_model(
        model,
        SAVED_MODEL_WEIGHTS_FILENAME,
        include_optimizer=False,
        save_format='h5')


if __name__ == '__main__':
    main()
