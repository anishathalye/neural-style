from stylize import *

import numpy as np
import scipy.misc

import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--style',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    content_image = imread(options.content)
    num_styles = len(options.styles)
    style_images = []
    for i in range(0,num_styles):
        style_images.append(imread(options.styles[i]))

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(0,num_styles):
        style_scale = STYLE_SCALE
        if options.style_scales != None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights == None:
        style_blend_weights = []
        # default is equal weights
        for i in range(0,num_styles):
            style_blend_weights.append(1.0/num_styles)

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    image = stylize(options.network, initial, content_image, style_images,
            options.iterations, options.content_weight, options.style_weight, style_blend_weights,
            options.tv_weight, options.learning_rate,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations)
    imsave(options.output, image)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

if __name__ == '__main__':
    main()
