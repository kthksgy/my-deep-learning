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
            if even_list[i + 1].lower() == 'true':
                ret[even_list[i]] = True
            elif even_list[i + 1].lower() == 'false':
                ret[even_list[i]] = False
            else:
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

    # モデル関連のオプション
    parser.add_argument(
        '-m', '--model',
        help='モデル名を指定します')
    parser.add_argument(
        '-mkw', '--model-kwargs', nargs='+',
        help='モデルに渡す追加パラメータを指定します')
    parser.add_argument(
        '-lm', '--list-models',
        help='モデル名の一覧を表示します。',
        action="store_true")

    # データ関連のオプション
    parser.add_argument(
        '-d', '--dataset',
        help='データセット名を指定します')
    parser.add_argument(
        '-bs', '--batch-size', type=int,
        help='バッチサイズを指定します')
    parser.add_argument(
        '-is', '--input-shape', type=int, nargs='+',
        help='入力のシェイプを指定します'
    )
    
    # 訓練のオプション
    parser.add_argument(
        '-e', '--epochs', type=int, default=100,
        help='総実行エポック数を指定します')
    
    # データオーギュメント
    parser.add_argument(
        '-augment',
        help='訓練データに対して予め用意されたデータオーギュメントを実施します。',
        action='store_true')

    # 一次元データ用のオプション
    parser.add_argument(
        '-c', '--compression',
        help='データに対して圧縮を行います',
        action='store_true')
    parser.add_argument(
        '-ckw', '--compression-kwargs', nargs='+',
        help='圧縮時のパラメータを指定します')
    parser.add_argument(
        '-il', '--input-length', type=int,
        help='系列の最大長さを指定します')
    parser.add_argument(
        '-check-length',
        help='指定したデータセットの最大長さを測定します',
        action="store_true")
    
    args = parser.parse_args()

    if args.list_models:
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
        'Python', sys.version.strip('\n'),
        'TensorFlow', tf.__version__,
        'Keras', keras.__version__
    ])))

    # TensorFlow 2ではない場合はEager Executionを有効にする
    if not tf.executing_eagerly():
        tf.enable_eager_execution()
    # 乱数種値を固定
    RANDOM_SEED = 0
    # TensorFlow, v1とv2で設定方法が違う
    if tf.__version__.startswith('1'):
        tf.random.set_random_seed(RANDOM_SEED)
    else:
        tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)  # Pythonのハッシュ関数
    np.random.seed(RANDOM_SEED)                   # numpy
    random.seed(RANDOM_SEED)                      # 標準のランダム
    ENVLOG.append('Random Seed\t%d' % RANDOM_SEED)
    # tf.debugging.set_log_device_placement(True)

    # floatの精度を指定
    DEFAULT_FLOATX = 'float16'
    keras.backend.set_floatx(DEFAULT_FLOATX)
    ENVLOG.append('Float X\t%s' % DEFAULT_FLOATX)

    # 動作フラグ
    DO_AUGMENTATION = args.augment
    DO_CHECK_LENGTH = args.check_length
    DO_COMPRESSION = args.compression

    # 定数定義
    # データセット関係の定数
    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch_size
    INPUT_SHAPE = tuple(args.input_shape) if not DO_COMPRESSION else tuple([args.input_shape[0]])
    #BATCH_SHAPE = tuple(tmp.extend(INPUT_SHAPE))
    ENVLOG.append('\t'.join(map(str, [
        'データセット', DATASET_NAME,
        'バッチサイズ', BATCH_SIZE,
        '入力シェイプ', INPUT_SHAPE
    ])))

    # 圧縮関連の定数
    COMPRESSION_KWARGS = list_to_dict(args.compression_kwargs)
    ENVLOG.append('\t'.join(map(str, [
        '圧縮方式', 'jpeg',
        '圧縮パラメータ', dict_to_oneline(COMPRESSION_KWARGS)
    ])))

    # モデル関係の定数
    MODEL_NAME = args.model
    MODEL_KWARGS = list_to_dict(args.model_kwargs)
    _tmp = MODEL_NAME.find('_')
    MODEL_FILENAME = MODEL_NAME if _tmp == -1 else MODEL_NAME[:_tmp]
    ENVLOG.append('\t'.join(map(str, [
        'モデル', MODEL_NAME,
        'モデルパラメータ', dict_to_oneline(MODEL_KWARGS),
    ])))

    # 訓練関連の定数
    EPOCHS = args.epochs

    # データセットの読み込み
    datasets, info = tfds_e.load(DATASET_NAME)
    NUM_CLASSES = info.features['label'].num_classes

    # 系列の長さチェックをやる場合はここで終了
    if DO_CHECK_LENGTH:
        print('系列の最大長さの計測を開始します。暫くお待ちください。')
        dataset_list = list(datasets.values())
        dataset = dataset_list.pop(0)
        for to_concat in dataset_list:
            dataset.concatenate(to_concat)
        # 前処理
        dataset = dataset.map(
            tfds_e.map_encode_jpeg(**COMPRESSION_KWARGS), NUM_CPUS)
        max_length = 0
        for image, label in dataset:
            max_length = max(max_length, len(image))
        print('最大長さ:', max_length)
        return

    # 結果出力用のファイルネーム
    RESULT_NAME_BASE = '{d}_{m}_{dt:%Y%m%d%H%M%S}'.format(
        d=args.dataset,
        m=args.model,
        dt=EXEC_DT
    )
    LOG_DIR = 'log_' + RESULT_NAME_BASE
    os.makedirs(LOG_DIR)
    RESULT_PATH = os.path.join(LOG_DIR, RESULT_NAME_BASE + '.csv')
    INFO_PATH = os.path.join(LOG_DIR, RESULT_NAME_BASE + '.tsv')
    SAVE_PATH = os.path.join(LOG_DIR, RESULT_NAME_BASE + '.h5')
    PLOT_PATH = os.path.join(LOG_DIR, RESULT_NAME_BASE + '.png')

    # データセットへの前処理
    for key in datasets:
        datasets[key] = datasets[key].map(
            lambda i, l: (i, keras.backend.one_hot(l, NUM_CLASSES)), NUM_CPUS
        )
        ENVLOG.append('データ前処理[%s]' % key)

        # シャッフルやデータオーギュメンテーション等は訓練用データのみに適用
        if key == 'train':
            datasets[key] = datasets[key].shuffle(BATCH_SIZE * 8)
            # データオーギュメンテーション
            if DO_AUGMENTATION:
                datasets[key] = datasets[key].map(
                    tfds_e.map_random_size_crop(0, 4, 0, 4, log=ENVLOG), NUM_CPUS)

        # 圧縮して時系列データへ
        if DO_COMPRESSION:
            datasets[key] = datasets[key].map(
                tfds_e.map_encode_jpeg(
                    quality=0, skip_header=True, log=ENVLOG), NUM_CPUS)
            datasets[key] = datasets[key].map(
                lambda image, label: (tf.cast(image, tf.int16), label), NUM_CPUS)
            datasets[key] = datasets[key].map(
                tfds_e.map_add_cls(cls_id=256, log=ENVLOG), NUM_CPUS)
            datasets[key] = datasets[key].map(
                tfds_e.map_shift_id(amount=1, log=ENVLOG), NUM_CPUS)
            # ミニバッチ化
            datasets[key] = datasets[key].padded_batch(
                BATCH_SIZE,
                (INPUT_SHAPE, ()),
                drop_remainder=True)
        else:
            # datasets[key] = datasets[key].map(
            #     lambda image, label: (tf.image.random_crop(image, [200, 200, 3]), label), NUM_CPUS)
            datasets[key] = datasets[key].map(
                lambda image, label: (tf.image.resize_with_crop_or_pad(image, INPUT_SHAPE[0], INPUT_SHAPE[1]), label), NUM_CPUS)
            datasets[key] = datasets[key].map(
                tfds_e.map_quantize_pixels(log=ENVLOG), NUM_CPUS)
            datasets[key] = datasets[key].map(
                lambda i, l: (tf.image.rgb_to_yuv(i), l), NUM_CPUS
            )
            # datasets[key] = datasets[key].map(
            #     tfds_e.map_blockwise_dct2(block_width=8, block_height=8, log=ENVLOG), NUM_CPUS
            # )
            datasets[key] = datasets[key].batch(BATCH_SIZE, drop_remainder=True)
        # 事前読み込みのパラメータ―1で自動調整モード
        datasets[key] = datasets[key].prefetch(-1)
    
    # INPUT_SHAPE = (INPUT_SHAPE[0] // 8, INPUT_SHAPE[1] // 8, 8 * 8 * INPUT_SHAPE[2])

    # モデルの読み込み
    if os.path.exists(SAVE_PATH):
        # 保存されたデータが有ればコンパイル無しで読み込む
        print('セーブファイルが存在するため読み込みを行います。')
        model = keras.models.load_model(
            SAVE_PATH,
            compile=False)
        print('読み込みが完了しました。')
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

    # OPTIMIZER = keras.optimizers.SGD(
    #     lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    OPTIMIZER = keras.optimizers.Adam()
    ENVLOG.append('\t'.join(map(str, [
        '最適化アルゴリズム', dict_to_oneline(OPTIMIZER.get_config())
    ])))

    LOSS = 'categorical_crossentropy'
    ENVLOG.append('\t'.join(map(str, [
        '損失関数', LOSS
    ])))
    def top_k_categorical_accuracy(y_true, y_pred, k=5):
        return keras.metrics.top_k_categorical_accuracy(
            keras.backend.cast(y_true, 'float32'),
            keras.backend.cast(y_pred, 'float32'),
            k=k
        )

    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=['categorical_accuracy', top_k_categorical_accuracy],
        experimental_run_tf_function=False)
    model.summary(print_fn=lambda s: ENVLOG.append('|' + s))
    model.summary()
    try:
        keras.utils.plot_model(
            model, PLOT_PATH, True, True, 'TB')
    except ImportError as e:
        print('graphvizとpydotが見つからないため、モデルの図の出力は出来ません。')
    except AssertionError as e:
        print('graphvizとpydotが見つからないため、モデルの図の出力は出来ません。')

    ENVLOG.append('訓練時のコールバック')
    callbacks = []
    callbacks.append(
        keras.callbacks.TerminateOnNaN()
    )
    callbacks.append(
        keras.callbacks.CSVLogger(
            RESULT_PATH, append=True)
    )
    # REVIEW: val_lossの監視だと期待したよりも早く停止してしまう
    # callbacks.append(
    #     keras.callbacks.EarlyStopping(
    #         'val_loss', 1e-4, 5, 1,
    #         restore_best_weights=True)
    # )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.9, verbose=1)
    )
    ENVLOG.append('\t'.join(map(str, [
        'ReduceLROnPlateau', 'val_loss, factor=0.9'
    ])))

    # 環境情報をファイルに書き込み
    with open(INFO_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ENVLOG))

    try:
        model.fit(
            datasets['train'],
            epochs=EPOCHS,
            verbose=2,
            callbacks=callbacks,
            validation_data=datasets[
                'validation' if 'validation' in datasets else
                'test'])
    except KeyboardInterrupt as e:
        pass
    finally:
        # モデルの保存
        keras.models.save_model(
            model,
            SAVE_PATH,
            include_optimizer=False,
            save_format='h5')
        print('処理が完了しました。')


if __name__ == '__main__':
    main()
