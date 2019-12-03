# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


class InferenceTimeMeasurement(keras.callbacks.Callback):
    def on_train_begin():
        pass