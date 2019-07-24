# Copyright (c) 2015-2019 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

# work-around for more recent versions of tensorflow
# https://github.com/tensorflow/tensorflow/issues/24496
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    if 'normalization' in data:
        # old format, for data where
        # MD5(imagenet-vgg-verydeep-19.mat) = 8ee3263992981a1d26e73b3ca028a123
        mean_pixel = np.mean(data['normalization'][0][0][0], axis=(0, 1))
    else:
        # new format, for data where
        # MD5(imagenet-vgg-verydeep-19.mat) = 106118b7cf60435e6d8e04f6a6dc3657
        mean_pixel = data['meta']['normalization'][0][0][0][0][2][0][0]
    weights = data['layers'][0]
    return weights, mean_pixel

def net_preloaded(weights, input_image, pooling):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            if isinstance(weights[i][0][0][0][0], np.ndarray):
                # old format
                kernels, bias = weights[i][0][0][0][0]
            else:
                # new format
                kernels, bias = weights[i][0][0][2][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel
