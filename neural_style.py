import vgg

import tensorflow as tf
import numpy as np
import scipy.misc as sm

import sys
import math

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
NOISE_RATIO = 0.0
ALPHA = 1.0 # weight of content loss
BETA = 1e4 # weight of style loss
LEARNING_RATE_INITIAL = 2e1
LEARNING_DECAY_BASE = 0.94
LEARNING_DECAY_STEPS = 100

def imread(path):
    return sm.imread(path).astype(np.float)

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    sm.imsave(path, img)

def main():
    content_path, style_path, width, style_scale = sys.argv[1:]
    width = int(width)
    style_scale = float(style_scale)

    content_image = imread(content_path)
    style_image = imread(style_path)

    if width <= 0:
        width = content_image.shape[1]

    content_aspect = (float(content_image.shape[0]) /
            content_image.shape[1]) # height / width
    new_shape = (int(math.floor(float(content_image.shape[0]) /
            content_image.shape[1] * width)), width)
    content_image = sm.imresize(content_image, new_shape)
    style_aspect = (float(style_image.shape[0]) /
            style_image.shape[1])
    if style_scale > 0:
        style_image_scaled = sm.imresize(style_image, style_scale)
        shape = style_image_scaled.shape
        if shape[0] >= new_shape[0] and shape[1] >= new_shape[1]:
            style_image = style_image_scaled
        else:
            style_scale = -1
    if style_scale <= 0:
        matched_height = int(math.ceil(new_shape[1] * style_aspect))
        if matched_height >= new_shape[0]:
            style_image = sm.imresize(style_image, (matched_height, new_shape[1]))
        else:
            matched_width = int(math.ceil(new_shape[0] / style_aspect))
            style_image = sm.imresize(style_image, (new_shape[0], matched_width))
    style_image = style_image[0:new_shape[0], 0:new_shape[1], :]
    assert content_image.shape == style_image.shape

    shape = (1,) + content_image.shape

    content_features = {}
    style_features = {}
    g = tf.Graph()
    with g.as_default():
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(VGG_PATH, image)

        with tf.Session() as sess:
            content_pre = np.array([vgg.preprocess(content_image, mean_pixel)])
            content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                    feed_dict={image: content_pre})

            style_pre = np.array([vgg.preprocess(style_image, mean_pixel)])
            for layer in STYLE_LAYERS:
                style_features[layer] = net[layer].eval(
                        feed_dict={image: style_pre})

    g = tf.Graph()
    with g.as_default():
        global_step = tf.Variable(0, trainable=False)
        noise = np.random.normal(size=shape, scale=np.std(content_image) * 0.1)
        content_pre = vgg.preprocess(content_image, mean_pixel)
        init = content_pre * (1 - NOISE_RATIO) + noise * NOISE_RATIO
        init = init.astype('float32')
        image = tf.Variable(init)
        net, _ = vgg.net(VGG_PATH, image)

        content_loss = tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER])
        style_losses = []
        for i in STYLE_LAYERS:
            layer = net[i]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats)

            match = style_features[i]
            match_feats = np.reshape(match, (-1, match.shape[3]))
            match_gram = np.matmul(match_feats.T, match_feats)

            style_losses.append(tf.nn.l2_loss(gram - match_gram) /
                    (4.0 * number ** 2 * (height * width) ** 2))
        style_loss = reduce(tf.add, style_losses) / len(style_losses)
        loss = ALPHA * content_loss + BETA * style_loss

        learning_rate = tf.train.exponential_decay(LEARNING_RATE_INITIAL,
                global_step, LEARNING_DECAY_STEPS, LEARNING_DECAY_BASE,
                staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(10000):
                print 'i = %d' % i
                imsave('%05d.jpg' % i, vgg.unprocess(
                        image.eval().reshape(shape[1:]), mean_pixel))
                train_step.run()


if __name__ == '__main__':
    main()
