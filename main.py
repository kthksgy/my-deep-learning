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
from data_augmentation import augment

# TensorFlow 2ではない場合は最初にEager Executionを有効にする
# tf.enable_eager_execution()


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
    # DEFAULT_FLOATX = 'float32'
    # keras.backend.set_floatx(DEFAULT_FLOATX)
    # ENVLOG.append('Float X\t%s' % DEFAULT_FLOATX)

    # 動作フラグ
    DO_AUGMENTATION = args.augment

    # 定数定義
    # データセット関係の定数
    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch_size
    INPUT_SHAPE = tuple(args.input_shape)
    #BATCH_SHAPE = tuple(tmp.extend(INPUT_SHAPE))
    ENVLOG.append('\t'.join(map(str, [
        'データセット', DATASET_NAME,
        'バッチサイズ', BATCH_SIZE,
        '入力シェイプ', INPUT_SHAPE
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
        # datasets[key] = datasets[key].map(
        #     lambda i, l: (i, keras.backend.one_hot(l, NUM_CLASSES)), NUM_CPUS
        # )
        ENVLOG.append('データ前処理[%s]' % key)
        datasets[key] = datasets[key].batch(BATCH_SIZE, drop_remainder=True)

        # シャッフルやデータオーギュメンテーション等は訓練用データのみに適用
        if key == 'train':
            datasets[key] = datasets[key].shuffle(BATCH_SIZE * 8)
            # データオーギュメンテーション
            if DO_AUGMENTATION:
                datasets[key] = augment(datasets[key], NUM_CPUS, INPUT_SHAPE[0], INPUT_SHAPE[1],
                    horizontal_flip=True, vertical_flip=False,
                    brightness_delta=0.1, hue_delta=0,
                    contrast_range=[0.9, 1.1], saturation_range=[0.9, 1.1],
                    width_shift=0.2, height_shift=0.2,
                    rotation=20)
        datasets[key] = datasets[key].map(
            lambda image, label: (tf.image.resize_with_crop_or_pad(image, INPUT_SHAPE[0], INPUT_SHAPE[1]), label), NUM_CPUS)
        # datasets[key] = datasets[key].map(
        #     tfds_e.map_quantize_pixels(log=ENVLOG), NUM_CPUS)
        # datasets[key] = datasets[key].batch(BATCH_SIZE, drop_remainder=True)
        # 事前読み込みのパラメータ―1で自動調整モード
        datasets[key] = datasets[key].prefetch(-1)

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
        model = load_model(
            datasets['train'].element_spec[0].shape[1:],
            NUM_CLASSES,
            batch_size=datasets['train'].element_spec[0].shape[0],
            **MODEL_KWARGS)
    if not model:
        print('モデルの読み込みに失敗しました。')
        return

    # OPTIMIZER = keras.optimizers.SGD(
    #     lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    OPTIMIZER = keras.optimizers.Adam()
    ENVLOG.append('\t'.join(map(str, [
        '最適化アルゴリズム', dict_to_oneline(OPTIMIZER.get_config())
    ])))

    LOSS = 'sparse_categorical_crossentropy'
    ENVLOG.append('\t'.join(map(str, [
        '損失関数', LOSS
    ])))

    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=['accuracy'],
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
