# coding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras

import tfds_e
from models.sample import sample
from models.xception import xception_keras


def main():
    print('Python Version: ', sys.version)
    print('TensorFlow Version: ', tf.__version__)
    print('Keras Version: ', keras.__version__)
    # TensorFlow 2ではない場合はEager Executionを有効にする
    if not tf.executing_eagerly():
        tf.enable_eager_execution()
    # 乱数種値を固定
    tf.random.set_random_seed(0)
    # tf.debugging.set_log_device_placement(True)

    DATASET_NAME = 'stl10'
    BATCH_SIZE = 500

    datasets, info = tfds_e.load(DATASET_NAME)

    for key in datasets:
        datasets[key] = datasets[key].map(tfds_e.map_quantize_pixels(), 16)
        datasets[key] = datasets[key].shuffle(BATCH_SIZE * 5)
        datasets[key] = datasets[key].batch(BATCH_SIZE, drop_remainder=True)
        datasets[key] = datasets[key].prefetch(-1)

    model = xception_keras(
        info.features['image'].shape,
        info.features['label'].num_classes
    )

    loss_object = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Nadam()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model.compile(optimizer=optimizer, loss=loss_object)
    model.summary()

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
        for image, label in datasets['train']:
            train_step(image, label)

        if('validation' in datasets):
            for test_image, test_label in datasets['validation']:
                test_step(test_image, test_label)

        if('test' in datasets):
            for test_image, test_label in datasets['test']:
                test_step(test_image, test_label)

        template = 'Epoch {:0%d}, Loss: {:.5f}, Accuracy: {:.4f}, Test Loss: {:.5f}, Test Accuracy: {:.4f}' % len(str(EPOCHS))
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

    model.save('{}.h5'.format(model.name))


if __name__ == '__main__':
    main()
