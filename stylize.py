import vgg

import tensorflow as tf
import numpy as np

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def stylize(network, initial, content, style, iterations,
        content_weight, style_weight, tv_weight,
        learning_rate, print_iter=None):
    shape = (1,) + content.shape
    style_shape = (1,) + style.shape
    content_features = {}
    style_features = {}

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(network, image)
        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                feed_dict={image: content_pre})

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net, _ = vgg.net(network, image)
        style_pre = np.array([vgg.preprocess(style, mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / (features.size)
            style_features[layer] = gram

    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 256 / 1000
        else:
            initial = np.array([vgg.preprocess(initial, mean_pixel)])
            initial = initial.astype('float32')
        image = tf.Variable(initial)
        net, _ = vgg.net(network, image)

        content_loss = content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                content_features[CONTENT_LAYER].size)
        style_losses = []
        for i in STYLE_LAYERS:
            layer = net[i]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / (size)
            style_gram = style_features[i]
            style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss = style_weight * reduce(tf.add, style_losses) / len(style_losses)
        tv_loss = tv_weight * (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) +
                tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]))
        loss = content_loss + style_loss + tv_loss

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(iterations):
                if print_iter is not None and i % print_iter == 0:
                    print '  content loss: %g' % (content_loss.eval())
                    print '    style loss: %g' % (style_loss.eval())
                    print '       tv loss: %g' % (tv_loss.eval())
                    print '    total loss: %g' % loss.eval()
                print 'Iteration %d/%d' % (i + 1, iterations)
                train_step.run()
            return vgg.unprocess(image.eval().reshape(shape[1:]), mean_pixel)
