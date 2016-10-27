# From official Dockerfile: https://github.com/tensorflow/tensorflow/blob/d44d271c9da4d244ce4b2ffaf808adbe4cff759d/tensorflow/tools/docker/Dockerfile
FROM tensorflow/tensorflow:latest

# install git add clone neural-style.git
RUN apt-get update
RUN apt-get install -y --no-install-recommends git
RUN git clone https://github.com/anishathalye/neural-style.git

# Pillow dependences
RUN apt-get install -y libffi-dev libssl-dev libtiff5-dev libjpeg8-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk
RUN pip install --trusted-host pypi.douban.com -i http://pypi.douban.com/simple/ -U pip
RUN pip install --trusted-host pypi.douban.com -i http://pypi.douban.com/simple/ -U Pillow
# RUN pip install --trusted-host pypi.douban.com -i http://pypi.douban.com/simple/ -U pyopenssl ndg-httpsclient pyasn1

# Pre-trained network, too large, use local volume instead.
RUN apt-get install -y wget
#RUN wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
#RUN mv imagenet-vgg-verydeep-19.mat neural-style

CMD ["/run_jupyter.sh"]
