# neural-style

An implementation of [neural style][paper] in
TensorFlow.

## Running

`python neural_style.py <content file> <style file> <output width> <style scale factor>`

If the width is set to -1, the output image is the same size as the content
image.

If the style scale factor is set to -1, the style image is scaled to the output
image (and then cropped).

## Example

Running it for 500-2000 iterations seems to produce nice results (using the
[Adam optimizer][adam]).

The following example was run for 1000 iterations to produce the result:

![output](examples/1-output.jpg)

These were the input images used (me sleeping at a hackathon and Starry Night):

![input-content](examples/1-content.jpg)

![input-style](examples/1-style.jpg)

## Requirements

* TensorFlow
* SciPy
* Pillow
* NumPy
* [Pre-trained VGG network][net]

## License

Copyright (c) 2015 Anish Athalye. Released under GPLv3. See
[LICENSE.md][license] for details.

[net]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[adam]: http://arxiv.org/abs/1412.6980
[license]: LICENSE.txt
