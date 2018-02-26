# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import neural_style_demo.losses as losses_utils
from neural_style_demo.nets import nets_factory
from neural_style_demo.preprocessing import preprocessing_factory

# 日志
tf_logger = logging.getLogger('tensorflow')
for _handler in tf_logger.handlers:
    tf_logger.removeHandler(_handler)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(filename)s - %(name)s - %(levelname)s - %(message)s", None)
)
tf_logger.addHandler(_handler)
tf.logging.set_verbosity(logging.INFO)

# global
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
            content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
            learning_rate, beta1, beta2, epsilon, pooling,
            print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    # global var
    model_name = "vgg_19"
    image_size = 224
    pretrained_model_file = network
    checkpoint_exclude_scopes = "vgg_19/fc"
    
    shape = (1, 224, 224, 3)
    content_features = {}
    style_features = [{} for _ in styles]
    
    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp
    
    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum
    
    # 计算 style features
    for i in range(len(styles)):
        _features = losses_utils.get_style_features(model_name=model_name,
                                                    style_image=styles[i],
                                                    image_size=image_size,
                                                    style_layers=STYLE_LAYERS,
                                                    naming="style-{}".format(i),
                                                    pretrained_model_file=pretrained_model_file,
                                                    checkpoint_exclude_scopes=checkpoint_exclude_scopes)
        
        for index, layer in enumerate(STYLE_LAYERS):
            style_features[i][layer] = _features[index]
    
    # 计算 content features
    content, _content_features = losses_utils.get_content_features(model_name=model_name,
                                                                   content_image=content,
                                                                   image_size=image_size,
                                                                   content_layers=CONTENT_LAYERS,
                                                                   naming="content",
                                                                   pretrained_model_file=pretrained_model_file,
                                                                   checkpoint_exclude_scopes=checkpoint_exclude_scopes)
    
    for index, layer in enumerate(CONTENT_LAYERS):
        content_features[layer] = _content_features[index]
    
    # 训练
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            
            # pre processed method
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                model_name,
                is_training=False
            )
            
            # make stylized image using bp
            if initial is None:
                initial_image = tf.random_normal(shape) * 0.256  # 4-D tensor, not need to be pre-processed
            else:
                initial_var = tf.Variable(initial)
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                pre_processed_init = sess.run(image_preprocessing_fn(initial_var, image_size, image_size))  # 3-D array
                initial_content_noise_coeff = 1.0 - initial_noiseblend
                initial_image = pre_processed_init * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (
                    1.0 - initial_content_noise_coeff)  # 4-D tensor
            
            images = tf.Variable(sess.run(initial_image), trainable=True)  # 4-D tensor
            network_fn = nets_factory.get_network_fn(model_name, num_classes=1, is_training=False)
            
            _, endpoint_dict = network_fn(images, spatial_squeeze=False)
            
            """Build Losses"""
            content_loss = losses_utils.content_loss(endpoint_dict, CONTENT_LAYERS, content_weight_blend,
                                                     content_weight, content_features, model_name)
            style_loss = losses_utils.style_loss(endpoint_dict, style_features, STYLE_LAYERS, styles,
                                                 style_layers_weights, style_weight, style_blend_weights, model_name)
            
            # todo maybe error; tv loss, use the unprocessed image
            tv_loss = losses_utils.total_variation_loss(
                tf.expand_dims(image_unprocessing_fn(tf.squeeze(images, [0])), axis=0), tv_weight
            )
            loss = content_loss + style_loss + tv_loss
            
            # optimizer setup
            train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss, var_list=[images])
            
            # init
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            init_func = losses_utils.get_init_fn(pretrained_model_file, checkpoint_exclude_scopes)
            init_func(sess)
            
            def print_progress():
                tf.logging.info("print progress")
            
            # optimization
            best_loss = float('inf')
            best = None
            
            # tensorboard log
            log_dir = "./logs"
            
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
                tf.gfile.MakeDirs(log_dir)
            
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            summary_writer.close()
            
            # train
            tf.logging.info('Optimization started...\n')
            if print_iterations and print_iterations != 0:
                print_progress()
            
            for i in range(iterations):
                tf.logging.info('Iteration %4d/%4d\n' % (i + 1, iterations))
                sess.run(train_step)
                tf.logging.info(
                    "sum of image is {}, loss is {}".format(sess.run(tf.reduce_sum(images)), sess.run(loss)))
                
                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()
                
                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = sess.run(loss)
                    tf.logging.info("step is {}, and loss is {}".format(i, this_loss))
                    
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = images
                    
                    if best is None:
                        best = images
                    
                    img_out = sess.run(image_unprocessing_fn(best[0, :]))  # 3-D
                    
                    if preserve_colors and preserve_colors is True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)
                        
                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB
                        
                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)
                        
                        # 2
                        styled_grayscale_yuv = np.array(
                            Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))
                        
                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))
                        
                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]
                        
                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
                    
                    yield ((None if last_step else i), img_out)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb
