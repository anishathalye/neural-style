# neural-style [![Build Status](https://github.com/anishathalye/neural-style/actions/workflows/ci.yml/badge.svg)](https://github.com/anishathalye/neural-style/actions/workflows/ci.yml)

An implementation of [neural style][paper] in TensorFlow.

This implementation is a lot simpler than a lot of the other ones out there,
thanks to TensorFlow's really nice API and [automatic differentiation][ad].

TensorFlow doesn't support [L-BFGS][l-bfgs] (which is what the original authors
used), so we use [Adam][adam]. This may require a little bit more
hyperparameter tuning to get nice results.

## Running

This project uses the [uv](https://docs.astral.sh/uv/) project and package manager; you can install it by [following these instructions](https://docs.astral.sh/uv/getting-started/installation/) (e.g., installing it with your system package manager, like `brew install uv`).

You also need to download [required data files](#requirements).

After that, you can run this program with:

```bash
uv run neural_style.py --content <content file> --styles <style file> --output <output file>
```

Run `uv run neural_style.py --help` to see a list of all options.

Use `--checkpoint-output` and `--checkpoint-iterations` to save checkpoint images.

Use `--iterations` to change the number of iterations (default 1000). For a 512×512 pixel content file, 1000 iterations take 90 seconds on an M3 MacBook Pro, and significantly less time with a more powerful (e.g., NVIDIA) GPU.

## Example 1

Running it for 500-2000 iterations seems to produce nice results. With certain
images or output sizes, you might need some hyperparameter tuning (especially
`--content-weight`, `--style-weight`, and `--learning-rate`).

The following example was run for 1000 iterations to produce the result (with
default parameters):

![output](examples/1-output.jpg)

These were the input images used (me sleeping at a hackathon and Starry Night):

![input-content](examples/1-content.jpg)

![input-style](examples/1-style.jpg)

## Example 2

The following example demonstrates style blending, and was run for 1000
iterations to produce the result (with style blend weight parameters 0.8 and
0.2):

![output](examples/2-output.jpg)

The content input image was a picture of the Stata Center at MIT:

![input-content](examples/2-content.jpg)

The style input images were Picasso's "Dora Maar" and Starry Night, with the
Picasso image having a style blend weight of 0.8 and Starry Night having a
style blend weight of 0.2:

![input-style](examples/2-style1.jpg)
![input-style](examples/2-style2.jpg)

## Tweaking

`--style-layer-weight-exp` command line argument could be used to tweak how "abstract"
the style transfer should be. Lower values mean that style transfer of a finer features
will be favored over style transfer of a more coarse features, and vice versa. Default
value is 1.0 - all layers treated equally. Somewhat extreme examples of what you can achieve:

![--style-layer-weight-exp 0.2](examples/tweaks/swe02.jpg)
![--style-layer-weight-exp 2.0](examples/tweaks/swe20.jpg)

(**left**: 0.2 - finer features style transfer; **right**: 2.0 - coarser features style transfer)

`--content-weight-blend` specifies the coefficient of content transfer layers. Default value -
1.0, style transfer tries to preserve finer grain content details. The value should be
in range [0.0; 1.0].

![--content-weight-blend 1.0](examples/tweaks/cwe10_default.jpg)
![--content-weight-blend 0.1](examples/tweaks/cwe01.jpg)

(**left**: 1.0 - default value; **right**: 0.1 - more abstract picture)

`--pooling` allows to select which pooling layers to use (specify either `max` or `avg`).
Original VGG topology uses max pooling, but the [style transfer paper][paper] suggests
replacing it with average pooling. The outputs are perceptually different, max pool in
general tends to have finer detail style transfer, but could have troubles at
lower-freqency detail level:

![--pooling max](examples/tweaks/swe14_pmax.jpg)
![--pooling avg](examples/tweaks/swe14_pavg.jpg)

(**left**: max pooling; **right**: average pooling)

`--preserve-colors` boolean command line argument adds post-processing step, which
combines colors from the original image and luma from the stylized image (YCbCr color
space), thus producing color-preserving style transfer:

![--pooling max](examples/tweaks/swe14_pmax.jpg)
![--pooling max](examples/tweaks/swe14_pmax_pcyuv.jpg)

(**left**: original stylized image; **right**: color-preserving style transfer)

## Requirements

### Data Files

* [Pre-trained VGG network][net] (SHA256 `abdb57167f82a2a1fbab1e1c16ad9373411883f262a1a37ee5db2e6fb0044695`) - put it in the top level of this repository, or specify its location using the `--network` option.

## Related Projects

See [here][lengstrom-fast-style-transfer] for an implementation of [fast
(feed-forward) neural style][fast-neural-style] in TensorFlow.

**[Try neural style](https://tenso.rs/demos/fast-neural-style/) client-side in
your web browser without installing any software (using
[TensorFire](https://tenso.rs/)).**

## Citation

If you use this implementation in your work, please cite the following:

```bibtex
@misc{athalye2015neuralstyle,
  author = {Anish Athalye},
  title = {Neural Style},
  year = {2015},
  howpublished = {\url{https://github.com/anishathalye/neural-style}},
}
```

## License

Copyright (c) Anish Athalye. Released under GPLv3. See
[LICENSE.txt][license] for details.

[net]: https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[adam]: http://arxiv.org/abs/1412.6980
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[lengstrom-fast-style-transfer]: https://github.com/lengstrom/fast-style-transfer
[fast-neural-style]: https://arxiv.org/pdf/1603.08155v1.pdf
[license]: LICENSE.txt
