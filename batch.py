from subprocess import call
import os

dir = os.getcwd()
styles = ['constable', 'dali', 'munch']
contents = ['sjc']

iterations = [500, 1000, 2000]
content_weight_blends = [0.1, 1.0]
style_layer_weight_exps = [0.1, 1.0, 3.0]
learning_rates = [1, 10]

for con in contents:
    content = dir + '\content\%s.jpg' % con
    for sty in styles:
        style = dir + '\styles\%s.jpg' % sty
        for iter in iterations:
            for cwb in content_weight_blends:
                for slwe in style_layer_weight_exps:
                    for lr in learning_rates:
                        if not os.path.exists(dir + '\outputs\%s\%s' % (con, sty)):
                            os.makedirs(dir + '\outputs\%s\%s' % (con, sty))
                        out_file = dir + '\outputs\%s\%s\%s_%s_%s_%s.jpg' % (con, sty, iter, cwb, slwe, lr)
                        cmd = "python neural_style.py --content %s --styles %s --output %s --style-layer-weight-exp %s --content-weight-blend %s --learning-rate %s --iterations %s" % (content, style, out_file, slwe, cwb, lr, iter)
                        call(cmd)