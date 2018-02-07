FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    git \
	python-pil \
    python-scipy \
    python-numpy \
    ffmpeg 

# Download neural-style
RUN git clone https://github.com/anishathalye/neural-style.git neural-style

# Download model
RUN cd neural-style; curl http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat -o imagenet-vgg-verydeep-19.mat; \
    echo "8ee3263992981a1d26e73b3ca028a123  imagenet-vgg-verydeep-19.mat" | md5sum -c -


# create volume for the images
RUN mkdir /images
VOLUME /images

# Prepare execution environment
WORKDIR /notebooks/neural-style/

CMD python neural_style.py --help


# docker build -t ahbrosha/neural-style-tf . && paplay /usr/share/sounds/gnome/default/alerts/sonar.ogg && notify-send -u urgent "Neural-Style"
# docker run --runtime=nvidia --rm -v $(pwd):/images ahbrosha/neural-style-tf python neural_style.py --content /images/brad_pitt.jpg --styles /images/picasso_selfport1907.jpg --output /images/profile.jpg --iterations 2000 --content-weight 10 --style-weight 1000 --checkpoint-output /images/output%s.jpg --checkpoint-iterations 100 --print-iterations 50
