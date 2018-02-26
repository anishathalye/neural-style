# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

from functools import reduce

import tensorflow as tf
import tensorflow.contrib.slim as slim

from neural_style_demo.nets import nets_factory
from neural_style_demo.preprocessing import preprocessing_factory


def get_init_fn(pretrained_model_file, checkpoint_exclude_scopes=None):
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % pretrained_model_file)
    
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    
    # todo error: when run in colab of google(tf 1.6.rc1-gpu), raise error
    # ValueError: The specified path: /content/vgg_19.ckpt is a file. Please specify only the path prefix to the checkpoint files.
    # but it work in my notebook(tf 1.6.rc1, only cpu)
    return slim.assign_from_checkpoint_fn(
        pretrained_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


def get_layer_name(model_scope, _layer_name):
    if _layer_name.find("relu") == -1:
        return None
    
    layer, index = _layer_name[4:].split("_")
    layer, conv_index = int(layer), int(index)
    
    conv_name = "conv{}".format(layer)
    return "{}/{}/{}_{}".format(model_scope, conv_name, conv_name, conv_index)


def gram(layer):
    shape = tf.shape(layer)
    num_images, width, height, num_filters = shape[0], shape[1], shape[2], shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    # todo not same as what in paper: grams = tf.matmul(filters, filters, transpose_a=True)
    return grams


def get_style_features(model_name, style_image, image_size, style_layers, naming, pretrained_model_file,
                       checkpoint_exclude_scopes=None):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=1,
            is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            model_name,
            is_training=False)
        
        # Add the batch dimension
        images = tf.expand_dims(image_preprocessing_fn(style_image, image_size, image_size), 0)
        
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in style_layers:
            _layer_name = get_layer_name(model_name, layer)
            feature = endpoints_dict[_layer_name]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)
        
        with tf.Session() as sess:
            # Restore variables for loss network.
            init_func = get_init_fn(pretrained_model_file, checkpoint_exclude_scopes)
            init_func(sess)
            
            # Make sure the 'generated' directory is exists.
            if tf.gfile.Exists('generated') is False:
                tf.gfile.MakeDirs('generated')
            
            # Indicate cropped style image path
            save_file = 'generated/target_style_' + naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)
            
            # save logs
            log_dir = './logs/'
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)
            
            train_writer = tf.summary.FileWriter(log_dir, sess.graph)
            train_writer.close()
            
            # Return the features those layers are use for measuring style loss.
            return sess.run(features)


def get_content_features(model_name, content_image, image_size, content_layers, naming, pretrained_model_file,
                         checkpoint_exclude_scopes=None):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=1,
            is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            model_name,
            is_training=False)
        
        # Add the batch dimension
        images = tf.expand_dims(image_preprocessing_fn(content_image, image_size, image_size), 0)
        
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in content_layers:
            _layer_name = get_layer_name(model_name, layer)
            feature = endpoints_dict[_layer_name]
            feature = tf.squeeze(feature, [0])  # remove the batch dimension
            features.append(feature)
        
        with tf.Session() as sess:
            # Restore variables for loss network.
            init_func = get_init_fn(pretrained_model_file, checkpoint_exclude_scopes)
            init_func(sess)
            
            # Make sure the 'generated' directory is exists.
            if tf.gfile.Exists('generated') is False:
                tf.gfile.MakeDirs('generated')
            
            # Indicate cropped style image path
            save_file = 'generated/target_content_' + naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target content pattern is saved to: %s.' % save_file)
            
            # save logs
            log_dir = './logs/'
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)
            
            train_writer = tf.summary.FileWriter(log_dir, sess.graph)
            train_writer.close()
            
            # Return the features those layers are use for measuring style loss.
            return sess.run([target_image, features])


def style_loss(endpoints_dict, style_features, style_layers, styles, style_layers_weights, style_weight,
               style_blend_weights, model_name):
    _style_loss = 0
    for i in range(len(styles)):
        style_losses = []
        for style_layer in style_layers:
            _current_layer = endpoints_dict[get_layer_name(model_name, style_layer)]
            _, height, width, number = map(lambda x: x.value, _current_layer.get_shape())
            style_gram = style_features[i][style_layer]
            style_losses.append(
                2 * style_layers_weights[style_layer] * tf.nn.l2_loss(gram(_current_layer) - style_gram) /
                tf.to_float(style_gram.size)
            )
        _style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
    
    return _style_loss


def content_loss(endpoints_dict, content_layers, content_weight_blend, content_weight, content_features, model_name):
    content_layers_weights = {'relu4_2': content_weight_blend, 'relu5_2': 1.0 - content_weight_blend}
    _content_loss = 0
    for content_layer in content_layers:
        _current_content_feature = endpoints_dict[get_layer_name(model_name, content_layer)]
        _target_feature = content_features[content_layer]
        _content_loss += 2 * content_weight * content_layers_weights[content_layer] * \
                         tf.nn.l2_loss(_target_feature - _current_content_feature) / \
                         tf.to_float(tf.size(_target_feature))  # remain the same as in the paper
    return _content_loss


def total_variation_loss(layer, tv_weight=1e2):
    shape = tf.shape(layer)
    _height, _width = shape[1], shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, _height - 1, -1, -1])) \
        - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, _width - 1, -1])) \
        - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    
    _loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return _loss * tv_weight
