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


def list_to_dict(even_list):
    ret = {}
    if even_list:
        for i in range(0, len(even_list), 2):
            try:
                ret[even_list[i]] = int(even_list[i + 1])
                continue
            except ValueError:
                pass
            try:
                ret[even_list[i]] = float(even_list[i + 1])
                continue
            except ValueError:
                pass
            try:
                ret[even_list[i]] = bool(even_list[i + 1])
                continue
            except ValueError:
                pass
            ret[even_list[i]] = even_list[i + 1]
    return ret


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
        '-il', '--inputlength', type=int,
        help='系列の最大長さを指定します')
    parser.add_argument(
        '-e', '--epochs', type=int, default=500,
        help='総実行エポック数を指定します')
    parser.add_argument(
        '-lm', '--listmodels',
        help='モデル名の一覧を表示します。',
        action="store_true")
    parser.add_argument(
        '-mk', '--modelkwargs', nargs='+',
        help='モデルに渡す追加パラメータを指定します')
    parser.add_argument(
        '-ck', '--compressionkwargs', nargs='+',
        help='圧縮時のパラメータを指定します'
    )
    parser.add_argument(
        '-checkmaxlength',
        help='指定したデータセットの最大長さを測定します',
        action="store_true")
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

    # 定数定義
    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch
    MODEL_NAME = args.model
    MODEL_KWARGS = list_to_dict(args.modelkwargs)
    _tmp = MODEL_NAME.find('_')
    MODEL_FILENAME = MODEL_NAME if _tmp == -1 else MODEL_NAME[:_tmp]
    EPOCHS = args.epochs
    INPUT_LENGTH = args.inputlength
    COMPRESSION_KWARGS = list_to_dict(args.compressionkwargs)

    # データセットの読み込み
    datasets, info = tfds_e.load(DATASET_NAME)

    if args.checkmaxlength:
        print('系列の最大長さの計測を開始します。暫くお待ちください。')
        dataset_list = list(datasets.values())
        dataset = dataset_list.pop(0)
        for to_concat in dataset_list:
            dataset.concatenate(to_concat)
        # 前処理
        dataset = dataset.map(tfds_e.map_encode_jpeg(**COMPRESSION_KWARGS), 16)
        max_length = 0
        for image, label in dataset:
            max_length = max(max_length, len(image))
        print('最大長さ:', max_length)
        return

    # データセットへの前処理
    for key in datasets:
        # 前処理
        datasets[key] = datasets[key].map(
            tfds_e.map_random_size_crop(0, 4, 0, 4), 16)
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
        model = keras.models.load_model(
            SAVED_MODEL_WEIGHTS_FILENAME,
            compile=False)
    else:
        # 新たなモデルを読み込む
        script = 'from models.%(filename)s\
            import model_%(modelname)s as load_model' % {
            'filename': MODEL_FILENAME,
            'modelname': MODEL_NAME
        }
        exec(script, globals())
        model = load_model(INPUT_SHAPE, NUM_CLASSES, **MODEL_KWARGS)
    if not model:
        print('モデルの読み込みに失敗しました。')
        return

    loss_object = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Nadam()

    model.compile(
        optimizer=optimizer, loss=loss_object,
        metrics=[keras.metrics.sparse_categorical_accuracy])
    model.summary()
    try:
        keras.utils.plot_model(
            model, MODEL_NAME_WITH_INPUT_SHAPE + '.png', True, True, 'TB')
    except ImportError as e:
        print('graphvizとpydotが見つからないため、モデルの図の出力は出来ません。')

    callbacks = []
    callbacks.append(
        keras.callbacks.TerminateOnNaN()
    )
    callbacks.append(
        keras.callbacks.CSVLogger(
            MODEL_NAME_WITH_INPUT_SHAPE + '.csv', append=True)
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.9, verbose=1)
    )
    # epochs_and_lrs = {
    #     0: 0.01,
    #     EPOCHS * 0.01: 0.005,
    #     EPOCHS * 0.03: 0.002,
    #     EPOCHS * 0.1: 0.001,
    #     EPOCHS * 0.9: 0.0001
    # }
    # lr_now = 0.001

    # def lr_schedule(epoch):
    #     global lr_now
    #     if epoch in epochs_and_lrs:
    #         lr_now = epochs_and_lrs[epoch]
    #         return epochs_and_lrs[epoch]
    #     else:
    #         return lr_now

    # callbacks.append(
    #     keras.callbacks.LearningRateScheduler(lr_schedule)
    # )

    model.fit(
        datasets['train'],
        epochs=EPOCHS,
        verbose=2,
        callbacks=callbacks,
        validation_data=datasets[
            'validation' if 'validation' in datasets else
            'test'])

    # モデルの保存
    keras.models.save_model(
        model,
        SAVED_MODEL_WEIGHTS_FILENAME,
        include_optimizer=False,
        save_format='h5')


if __name__ == '__main__':
    main()
