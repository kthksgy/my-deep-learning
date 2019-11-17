# -*- coding: utf-8 -*-
"""TensorFlow Kerasのモデル"""
from tensorflow import keras


def _se(x):
    """Squeeze-And-Excite.
    """
    input_channels = int(x.shape[-1])
    x_old = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(input_channels, activation='relu')(x)
    x = keras.layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = keras.layers.Reshape((1, 1, input_channels))(x)
    x = keras.layers.Multiply()([x, x_old])
    return x


def __re(x):
    """ReLU6
    """
    return keras.backend.relu(x, max_value=6.0)


def __hs(x):
    """h-swish nonlinearity.
    """
    return x * keras.backend.relu(x + 3.0, max_value=6.0) / 6.0


def _activation(nl):
    if nl == 'HS':
        return keras.layers.Activation(__hs)
    elif nl == 'RE':
        return keras.layers.Activation(__re)
    else:
        return keras.layers.ReLU()


def _conv_bn(x, filters: int, kernel_size: int, strides: int, nl: str):
    """Convolution Block
    This function defines a 2D convolution operation with BN and activation.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """

    x = keras.layers.Conv2D(filters, kernel_size, strides, 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = _activation(nl)(x)
    return x

def _bottleneck(
    x, filters: int, kernel_size: int, strides: int,
    expansion_size: int, se: bool, nl: str, alpha=1.0):
    """Bottleneck
    """
    input_shape = keras.backend.int_shape(x)
    x_old = x

    x = _conv_bn(x, expansion_size, 1, 1, nl)

    x = keras.layers.DepthwiseConv2D(kernel_size, strides, 'same', 1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = _activation(nl)(x)
    if se:
        x = _se(x)
    x = keras.layers.Conv2D(int(alpha * filters), 1, 1, 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    if strides == 1 and input_shape[3] == filters:
        x = keras.layers.Add()([x, x_old])
    return x


def model_mobilenet_v3(input_shape: tuple, num_classes: int) -> keras.Model:
    """build MobileNetV3 Large.
    # Arguments
        plot: Boolean, weather to plot model.
    # Returns
        model: Model, model.
    """
    inputs = keras.layers.Input(shape=input_shape)

    x = _conv_bn(inputs, 16, 3, 2, 'HS')

    # #out, kernel_size, strides, exp_size, SE, NL
    x = _bottleneck(x, 16, 3, 1, 16, False, 'RE')
    x = _bottleneck(x, 24, 3, 2, 64, False, 'RE')
    x = _bottleneck(x, 24, 3, 1, 72, False, 'RE')
    x = _bottleneck(x, 40, 5, 2, 72, True, 'RE')
    x = _bottleneck(x, 40, 5, 1, 120, True, 'RE')
    x = _bottleneck(x, 40, 5, 1, 120, True, 'RE')
    x = _bottleneck(x, 80, 3, 2, 240, False, 'HS')
    x = _bottleneck(x, 80, 3, 1, 200, False, 'HS')
    x = _bottleneck(x, 80, 3, 1, 184, False, 'HS')
    x = _bottleneck(x, 80, 3, 1, 184, False, 'HS')
    x = _bottleneck(x, 112, 3, 1, 480, True, 'HS')
    x = _bottleneck(x, 112, 3, 1, 672, True, 'HS')
    x = _bottleneck(x, 160, 5, 2, 672, True, 'HS')
    x = _bottleneck(x, 160, 5, 1, 960, True, 'HS')
    x = _bottleneck(x, 160, 5, 1, 960, True, 'HS')

    x = _conv_bn(x, 960, 1, 1, 'HS')
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, 960))(x)

    x = keras.layers.Conv2D(1280, (1, 1), padding='same')(x)
    x = _activation('HS')(x)

    x = keras.layers.Conv2D(num_classes, 1, 1, 'same', activation='softmax')(x)
    x = keras.layers.Reshape((num_classes,))(x)
    outputs = x
    return keras.Model(inputs=inputs, outputs=outputs, name='MobileNet V3')
