# coding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras

from model_loader import ModelLoader
import tfds_e


def main():
    print('Python Version: ', sys.version)
    print('TensorFlow Version: ', tf.__version__)
    print('Keras Version: ', keras.__version__)
    # tf.debugging.set_log_device_placement(True)

    DATASET_NAME = 'cifar10'
    DATA_DIR = r'~/.datasets'
    BATCH_SIZE = 1000

    DO_DCT = False

    model_loader = ModelLoader()

    datasets, info = tfds_e.load(DATASET_NAME, data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    width, height, channels = info.features['image'].shape
    shape = info.features['image'].shape if not DO_DCT else (height, width * channels)

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    def augment(image, label):
        image = tf.image.random_crop(image, shape)

    def dct(image, label):
        transposed = tf.transpose(image, (0, 1, 3, 2))
        dcted = tf.signal.dct(transposed, norm='ortho')
        reshaped = tf.reshape(dcted, (BATCH_SIZE, height, -1))
        return reshaped, label

    for key in datasets:
        datasets[key] = datasets[key].map(scale, num_parallel_calls=16)
        if(DO_DCT):
            datasets[key] = datasets[key].map(dct, num_parallel_calls=16)

    model = model_loader.load(
        shape=shape,
        name="xception",
        classes=info.features['label'].num_classes
    )
    model.summary()

    loss_object = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Nadam()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model.compile(optimizer=optimizer, loss=loss_object)

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
