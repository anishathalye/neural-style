# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from neural_style_demo.neural_style import main as neural_style_main


def demo():
    args = ["--content", "examples/1-content.jpg",
            "--styles", "examples/1-style.jpg", "--output", "xsplus-1000.jpg", "--iterations",
            "1000", "--network", "/home/frkhit/Downloads/AI/pre-trained-model/vgg_19.ckpt",
            "--checkpoint-iterations", "10", "--checkpoint-output", "tmp-tv-%s.jpg"]
    neural_style_main(args)


if __name__ == '__main__':
    demo()
