# -*- coding: UTF-8 -*-
from __future__ import (
    absolute_import, division,
    print_function, unicode_literals)

import argparse
import datetime
import glob
import multiprocessing
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


def dict_to_oneline(to_oneline: dict) -> str:
    return ', '.join([
        '%s=%s' % (key, value) for key, value in to_oneline.items()])


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
        '-mkw', '--modelkwargs', nargs='+',
        help='モデルに渡す追加パラメータを指定します')
    parser.add_argument(
        '-ckw', '--compressionkwargs', nargs='+',
        help='圧縮時のパラメータを指定します'
    )
    parser.add_argument(
        '-augment',
        help='訓練データに対して予め用意されたデータオーギュメントを実施します。',
        action='store_true'
    )
    parser.add_argument(
        '-checkmaxlength',
        help='指定したデータセットの最大長さを測定します',
        action="store_true")
    args = parser.parse_args()

    if args.listmodels:
        print('モデル名: ', get_model_names())
        return

    NUM_CPUS = multiprocessing.cpu_count()
    EXEC_DT = datetime.datetime.now()

    # バージョン表示
    print('Python Version: ', sys.version)
    print('TensorFlow Version: ', tf.__version__)
    print('Keras Version: ', keras.__version__)
    ENVLOG = list()
    ENVLOG.append('\t'.join(map(str, [
        'Python', sys.version,
        'TensorFlow', tf.__version__,
        'Keras', keras.__version__
    ])))

    # TensorFlow 2ではない場合はEager Executionを有効にする
    if not tf.executing_eagerly():
        tf.enable_eager_execution()
    # 乱数種値を固定
    RANDOM_SEED = 0
    tf.random.set_random_seed(RANDOM_SEED)        # TensorFlow
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)  # Pythonのハッシュ関数
    np.random.seed(RANDOM_SEED)                   # numpy
    random.seed(RANDOM_SEED)                      # 標準のランダム
    ENVLOG.append('Random Seed\t%d' % RANDOM_SEED)
    # tf.debugging.set_log_device_placement(True)

    # 定数定義
    # データセット関係の定数
    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch
    INPUT_LENGTH = args.inputlength
    ENVLOG.append('\t'.join(map(str, [
        'データセット', DATASET_NAME,
        'バッチサイズ', BATCH_SIZE,
        '最大入力長', INPUT_LENGTH
    ])))
    AUGMENT = args.augment

    # 圧縮関連の定数
    COMPRESSION_KWARGS = list_to_dict(args.compressionkwargs)
    ENVLOG.append('\t'.join(map(str, [
        '圧縮方式', 'jpeg',
        '圧縮パラメータ', dict_to_oneline(COMPRESSION_KWARGS)
    ])))

    # モデル関係の定数
    MODEL_NAME = args.model
    MODEL_KWARGS = list_to_dict(args.modelkwargs)
    _tmp = MODEL_NAME.find('_')
    MODEL_FILENAME = MODEL_NAME if _tmp == -1 else MODEL_NAME[:_tmp]
    ENVLOG.append('\t'.join(map(str, [
        'モデル', MODEL_NAME,
        'モデルパラメータ', dict_to_oneline(MODEL_KWARGS),
    ])))

    # 訓練関連の定数
    EPOCHS = args.epochs

    # 結果出力用のファイルネーム
    RESULT_NAME_BASE = '{d}_{m}_{dt:%Y%m%d%H%M%S}'.format(
        d=args.dataset,
        m=args.model,
        dt=EXEC_DT
    )
    RESULT_FILENAME = RESULT_NAME_BASE + '.csv'
    INFO_FILENAME = RESULT_NAME_BASE + '.tsv'
    SAVE_FILENAME = RESULT_NAME_BASE + '.h5'
    PLOT_FILENAME = RESULT_NAME_BASE + '.png'

    # データセットの読み込み
    datasets, info = tfds_e.load(DATASET_NAME)

    # 系列の長さチェックをやる場合はここで終了
    if args.checkmaxlength:
        print('系列の最大長さの計測を開始します。暫くお待ちください。')
        dataset_list = list(datasets.values())
        dataset = dataset_list.pop(0)
        for to_concat in dataset_list:
            dataset.concatenate(to_concat)
        # 前処理
        dataset = dataset.map(tfds_e.map_encode_jpeg(**COMPRESSION_KWARGS), NUM_CPUS)
        max_length = 0
        for image, label in dataset:
            max_length = max(max_length, len(image))
        print('最大長さ:', max_length)
        return

    # データセットへの前処理
    for key in datasets:
        ENVLOG.append('データ前処理[%s]' % key)
        # データオーギュメンテーション
        if AUGMENT and key == 'train':
            datasets[key] = datasets[key].map(
                tfds_e.map_random_size_crop(0, 4, 0, 4), NUM_CPUS)

        # 圧縮して時系列データへ
        datasets[key] = datasets[key].map(
            tfds_e.map_encode_jpeg(
                quality=0, skip_header=True, log=ENVLOG), NUM_CPUS)
        datasets[key] = datasets[key].map(
            lambda image, label: (tf.cast(image, tf.int16), label), NUM_CPUS)
        datasets[key] = datasets[key].map(
            tfds_e.map_add_cls(cls_id=256, log=ENVLOG), NUM_CPUS)
        datasets[key] = datasets[key].map(
            tfds_e.map_shift_id(amount=1, log=ENVLOG), NUM_CPUS)

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

    # モデルの読み込み
    if os.path.exists(SAVE_FILENAME):
        # 保存されたデータが有ればコンパイル無しで読み込む
        model = keras.models.load_model(
            SAVE_FILENAME,
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
    model.summary(print_fn=lambda s: ENVLOG.append('|' + s))
    try:
        keras.utils.plot_model(
            model, PLOT_FILENAME, True, True, 'TB')
    except ImportError as e:
        print('graphvizとpydotが見つからないため、モデルの図の出力は出来ません。')

    ENVLOG.append('訓練時のコールバック')
    callbacks = []
    callbacks.append(
        keras.callbacks.TerminateOnNaN()
    )
    callbacks.append(
        keras.callbacks.CSVLogger(
            RESULT_FILENAME, append=True)
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.9, verbose=1)
    )
    ENVLOG.append('\t'.join(map(str, [
        'ReduceLROnPlateau', 'val_loss, factor=0.9'
    ])))

    # 環境情報をファイルに書き込み
    with open(INFO_FILENAME, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ENVLOG))

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
        SAVE_FILENAME,
        include_optimizer=False,
        save_format='h5')


if __name__ == '__main__':
    main()
