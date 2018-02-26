## 0.配置环境并授权
```
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```

## 1.挂载google drive
```
!mkdir -p drive
!google-drive-ocamlfuse drive

```

## 2.安装neural_style_demo环境
```
!cp /content/drive/ai/neural_style_demo.tar.gz /usr/local/lib/python3.6/dist-packages/
! cd /usr/local/lib/python3.6/dist-packages/ && tar -xzvf neural_style_demo.tar.gz
! cd /content
! ls /usr/local/lib/python3.6/dist-packages/ | grep neural
!python -m neural_style_demo
```

## 3.准备vgg19
```
!wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
!tar -xzvf vgg_19_2016_08_28.tar.gz
!ls
```

## 4.执行代码

``` 
from neural_style_demo.neural_style import main as neural_style_main


def demo():
    args = ["--content", "/usr/local/lib/python3.6/dist-packages/neural_style_demo/examples/1-content.jpg",
            "--styles", "/usr/local/lib/python3.6/dist-packages/neural_style_demo/examples/1-style.jpg", "--output", "/content/xsplus-1000.jpg", "--iterations",
            "1000", "--network", "/content/vgg_19.ckpt",
            "--checkpoint-iterations", "10", "--checkpoint-output", "/content/tmp-tv-%s.jpg"]
    neural_style_main(args)


demo()

```

出错:

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-29-b803c1b9c409> in <module>()
     10 
     11 
---> 12 demo()

<ipython-input-29-b803c1b9c409> in demo()
      7             "1000", "--network", "/content/vgg_19.ckpt",
      8             "--checkpoint-iterations", "10", "--checkpoint-output", "/content/tmp-tv-%s.jpg"]
----> 9     neural_style_main(args)
     10 
     11 

/usr/local/lib/python3.6/dist-packages/neural_style_demo/neural_style.py in main(args)
    178         pooling=options.pooling,
    179         print_iterations=options.print_iterations,
--> 180         checkpoint_iterations=options.checkpoint_iterations
    181     ):
    182         output_file = None

/usr/local/lib/python3.6/dist-packages/neural_style_demo/stylize.py in stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations, content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight, learning_rate, beta1, beta2, epsilon, pooling, print_iterations, checkpoint_iterations)
     77                                                     naming="style-{}".format(i),
     78                                                     pretrained_model_file=pretrained_model_file,
---> 79                                                     checkpoint_exclude_scopes=checkpoint_exclude_scopes)
     80 
     81         for index, layer in enumerate(STYLE_LAYERS):

/usr/local/lib/python3.6/dist-packages/neural_style_demo/losses.py in get_style_features(model_name, style_image, image_size, style_layers, naming, pretrained_model_file, checkpoint_exclude_scopes)
     97                 break
     98         if not excluded:
---> 99             variables_to_restore.append(var)
    100 
    101     return assign_from_checkpoint_fn(

/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/framework/python/ops/variables.py in callback(session)
    688     saver = tf_saver.Saver(var_list, reshape=reshape_variables)
    689     def callback(session):
--> 690       saver.restore(session, model_path)
    691     return callback
    692   else:

/usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py in restore(self, sess, save_path)
   1754       raise ValueError("The specified path: %s is a file."
   1755                        " Please specify only the path prefix"
-> 1756                        " to the checkpoint files." % save_path)
   1757     logging.info("Restoring parameters from %s", save_path)
   1758     if context.in_graph_mode():

ValueError: The specified path: /content/vgg_19.ckpt is a file. Please specify only the path prefix to the checkpoint files.
```