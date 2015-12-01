from stylize import *

import numpy as np
import scipy.misc as sm

import math
from argparse import ArgumentParser

# defaults
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
ITERATIONS = 1000

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--style',
            dest='style', help='style image',
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
    parser.add_argument('--style-scale', type=float,
            dest='style_scale', help='style scale',
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
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--print-iter', type=int,
            dest='print_iter', help='print statistics after these many iterations',
            metavar='PRINT_ITER')
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    width = options.width
    style_scale = options.style_scale

    content_image = imread(options.content)
    style_image = imread(options.style)

    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = sm.imresize(content_image, new_shape)
    if style_scale is not None:
        style_image = sm.imresize(style_image, style_scale)

    initial = options.initial
    if initial is not None:
        initial = sm.imresize(imread(initial), content_image.shape[:2])

    image = stylize(options.network, initial, content_image, style_image,
            options.iterations, options.content_weight, options.style_weight,
            options.tv_weight, options.learning_rate, print_iter=options.print_iter)
    imsave(options.output, image)

def imread(path):
    return sm.imread(path).astype(np.float)

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    sm.imsave(path, img)

if __name__ == '__main__':
    main()
