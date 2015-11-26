import tensorflow as tf
import numpy as np
import scipy.io as sio

def _conv_layer(weights, bias):
    def _make_layer(input):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=[1, 1, 1, 1],
                padding='SAME')
        return tf.nn.bias_add(conv, bias)
    return _make_layer

def _pool_layer():
    def _make_layer(input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return _make_layer

def _add_layer(input_image, layers, func):
    if not layers:
        new = func(input_image)
    else:
        new = func(layers[-1])
    layers.append(new)

def net(data_path, input_image):
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    ]


    data = sio.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    constants = data['layers'][0]

    net = []
    for i, kind in enumerate(layers):
        short = kind[:4]
        if short == 'conv':
            weights = constants[i][0][0][0][0][0]
            # in matconvnet, weights are [width, height, depth, num_filters]
            # but in tensorflow, [height, width, in_channels, out_channels]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = constants[i][0][0][0][0][1].reshape(-1)
            new = _conv_layer(weights, bias)
        elif short == 'relu':
            new = tf.nn.relu
        elif short == 'pool':
            new = _pool_layer()
        else:
            raise ValueError('invalid layer type: %s' % kind)
        _add_layer(input_image, net, new)

    assert len(layers) == len(net)

    return dict(zip(layers, net)), mean_pixel

def preprocess(image, mean_pixel):
    return image - mean_pixel

def unprocess(image, mean_pixel):
    image = image + mean_pixel
    return image
