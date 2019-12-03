# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from scipy.signal import convolve2d


def augment(ds: tf.data.Dataset, num_parallel_calls: int, target_height: int, target_width: int,
    validation=False, horizontal_flip=False, vertical_flip=False,
    brightness_delta=0, hue_delta=0,
    contrast_range=None, saturation_range=None,
    width_shift=0.0, height_shift=0.0,
    rotation=0.0, jpeg2000=False):
    if isinstance(ds.element_spec, dict):
        print('(image, label)のタプルが出力されるtf.data.Datasetを渡してください．')
        return ds
    
    batch_shape = tuple(ds.element_spec[0].shape)

    ds = ds.map(lambda i, l: (
        tf.image.resize(i, [target_height, target_width], preserve_aspect_ratio=True), l), num_parallel_calls)
    
    if validation:
        if jpeg2000:
            ds = ds.map(lambda i, l: (tf.image.rgb_to_yuv(i), l), num_parallel_calls)
            ds = ds.map(lambda i, l: tf.py_function(_jpeg2000_batch, [i, l], [tf.float32, tf.int64]), num_parallel_calls)
        return ds

    if horizontal_flip:
        ds = ds.map(lambda i, l: (tf.image.random_flip_left_right(i), l), num_parallel_calls)
    
    if vertical_flip:
        ds = ds.map(lambda i, l: (tf.image.random_flip_up_down(i), l), num_parallel_calls)
    
    if brightness_delta > 0:
        ds = ds.map(lambda i, l: (tf.image.random_brightness(i, brightness_delta), l), num_parallel_calls)

    if hue_delta > 0:
        ds = ds.map(lambda i, l: (tf.image.random_hue(i, hue_delta), l), num_parallel_calls)
    
    if contrast_range:
        if isinstance(contrast_range, list) or isinstance(contrast_range, tuple):
            contrast_lower, contrast_upper = contrast_range
        elif contrast_range > 0:
            contrast_lower, contrast_upper = 1.0 - contrast_range, 1.0 + contrast_range
        ds = ds.map(lambda i, l: (tf.image.random_contrast(i, contrast_lower, contrast_upper), l), num_parallel_calls)
    
    if saturation_range:
        if isinstance(saturation_range, list) or isinstance(saturation_range, tuple):
            saturation_lower, saturation_upper = saturation_range
        elif saturation_range > 0:
            saturation_lower, saturation_upper = 1.0 - saturation_range, 1.0 + saturation_range
        ds = ds.map(lambda i, l: (tf.image.random_saturation(i, saturation_lower, saturation_upper), l), num_parallel_calls)

    if height_shift > 0 or width_shift > 0:
        if isinstance(height_shift, float):
            height_shift = int(target_height * height_shift)
            pass
        if isinstance(width_shift, float):
            width_shift = int(target_width * width_shift)
            pass
        ds = ds.map(lambda i, l: (tfa.image.translate(i, [width_shift, height_shift]), l), num_parallel_calls)
    
    if rotation > 0:
        ds = ds.map(lambda i, l: (tfa.image.rotate(i, tf.random.uniform([batch_shape[0]], np.pi * -rotation / 180, np.pi * rotation / 180)), l), num_parallel_calls)
    
    # ds = ds.map(lambda i, l: (tf.image.resize_with_crop_or_pad(i, target_height, target_width), l), num_parallel_calls)
    if jpeg2000:
        ds = ds.map(lambda i, l: (tf.image.rgb_to_yuv(i), l), num_parallel_calls)
        ds = ds.map(lambda i, l: tf.py_function(_jpeg2000_batch, [i, l], [tf.float32, tf.int64]), num_parallel_calls)

    return ds

# 
jpeg2000_e = 5.5
jpeg2000_m = 8
DELTA = np.power(2, 8 - jpeg2000_e) * (1 + jpeg2000_m / np.power(2, 11))

LCOEF_9_7 = np.asarray([
    0.6029490182363579,
    0.2668641184428723,
    -0.07822326652898785,
    -0.01686411844287495,
    0.02674875741080976
])
LCOEF_9_7 = np.append(np.flip(LCOEF_9_7)[:-1], LCOEF_9_7)
HCOEF_9_7 = np.asarray([
    1.115087052456994,
    -0.5912717631142470,
    -0.05754352622849957,
    0.09127176311424948
])
HCOEF_9_7 = np.append(np.flip(HCOEF_9_7)[:-1], HCOEF_9_7)
LCOEF_5_3 = np.asarray([
    6 / 8,
    2 / 8,
    -1 / 8,
])
LCOEF_5_3 = np.append(np.flip(LCOEF_5_3)[:-1], LCOEF_5_3)
HCOEF_5_3 = np.asarray([
    1,
    -1 / 2
])
HCOEF_5_3 = np.append(np.flip(HCOEF_5_3)[:-1], HCOEF_5_3)

def _jpeg2000_batch(x, y):
    return np.array(list(map(_jpeg2000, x))), y
    
def _jpeg2000(x):
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)
    ret = np.zeros(x.shape)
    src = x
    w = x.shape[1]
    h = x.shape[0]
    for l in range(3):
        for i, c in enumerate(np.transpose(src, (2, 0, 1))):
            ll = convolve2d(c, np.expand_dims(LCOEF_9_7, 0), 'same', 'symm')
            ll = convolve2d(ll, np.expand_dims(LCOEF_9_7, -1), 'same', 'symm')
            ret[0:h//2, 0:w//2, i] = ll[::2, ::2]

            hl = convolve2d(c, np.expand_dims(LCOEF_9_7, 0), 'same', 'symm')
            hl = convolve2d(hl, np.expand_dims(HCOEF_9_7, -1), 'same', 'symm')
            ret[0:h//2, w//2:w, i] = hl[::2, 1::2]

            lh = convolve2d(c, np.expand_dims(HCOEF_9_7, 0), 'same', 'symm')
            lh = convolve2d(lh, np.expand_dims(LCOEF_9_7, -1), 'same', 'symm')
            ret[h//2:h, 0:w//2, i] = lh[1::2, ::2]

            hh = convolve2d(c, np.expand_dims(HCOEF_9_7, 0), 'same', 'symm')
            hh = convolve2d(hh, np.expand_dims(HCOEF_9_7, -1), 'same', 'symm')
            ret[h//2:h, w//2:w, i] = hh[1::2, 1::2]

        src = np.copy(ret[0:h//2, 0:w//2])
        h = h // 2
        w = w // 2
    ret = np.sign(ret) * (np.abs(ret) // DELTA)
    return ret