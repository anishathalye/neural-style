# neural-style

An implementation of [neural style][paper] in TensorFlow.

This implementation is a lot simpler than a lot of the other ones out there,
thanks to TensorFlow's really nice API and [automatic differentiation][ad].

The algorithm seems to be working all right, but the results aren't always as
good as some of the other implementations. This may be due to the optimization
algorithm used - TensorFlow doesn't support [L-BFGS][l-bfgs], so we use
[Adam][adam]. It may be due to the parameters used. Or it may be a bug in the
code... I don't know yet. **Any help improving the code would be much
appreciated!**

Also, TensorFlow seems to be [slower][tensorflow-benchmarks] than a lot of the
other deep learning frameworks out there. I'm sure this implementation could be
improved, but it would probably take improvements in TensorFlow itself as well
to get it to operate at the same speed as other implementations.

## Running

`python neural_style.py <content file> <style file> <output width> <style scale factor>`

(The CLI could use some work... I'll fix it soon.)

If the width is set to -1, the output image is the same size as the content
image.

If the style scale factor is set to -1, the style image is scaled to the output
image (and then cropped).

## Example

Running it for 500-2000 iterations seems to produce nice results.

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
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[adam]: http://arxiv.org/abs/1412.6980
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[tensorflow-benchmarks]: https://github.com/soumith/convnet-benchmarks
[license]: LICENSE.txt
