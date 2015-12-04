import vgg

import tensorflow as tf
import numpy as np
import sys

from sys import stderr

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def stylize(network, initial, content, style, iterations,
        content_weight, style_weight, tv_weight,
        learning_rate, print_iterations=None, checkpoint_iterations=None, target_loss=0.0):
    shape = (1,) + content.shape
    style_shape = (1,) + style.shape
    content_features = {}
    style_features = {}

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(network, image)
        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                feed_dict={image: content_pre})

    # compute style features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net, _ = vgg.net(network, image)
        style_pre = np.array([vgg.preprocess(style, mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, mean_pixel)])
            initial = initial.astype('float32')
        image = tf.Variable(initial)
        net, _ = vgg.net(network, image)

        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                content_features[CONTENT_LAYER].size)
        # style loss
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) /
                    style_gram.size)
        style_loss = style_weight * reduce(tf.add, style_losses)
        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        def print_progress(i, last=False):
            if print_iterations is not None:
                if i is not None and i % print_iterations == 0 or last:
                    print >> stderr, '  content loss: %g' % content_loss.eval()
                    print >> stderr, '    style loss: %g' % style_loss.eval()
                    print >> stderr, '       tv loss: %g' % tv_loss.eval()
                    print >> stderr, '    total loss: %g' % loss.eval()

        # optimization
        assert checkpoint_iterations is not None # ugly hack cause lazy
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            i = 0
            while i < iterations and best_loss > target_loss:
                print_progress(i)
                print >> stderr, 'Iteration %d/%d' % (i + 1, iterations)
                train_step.run()
                if checkpoint_iterations is not None:
                    if i % checkpoint_iterations == 0:
                        this_loss = loss.eval()
                        if this_loss < best_loss:
                            best_loss = this_loss
                            best = image.eval()
                print_progress(None, i == iterations - 1)
                i += 1
            return (vgg.unprocess(best).reshape(shape[1:]), mean_pixel), best_loss)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
