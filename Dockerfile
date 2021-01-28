FROM python:3.7-buster

RUN apt-get update

COPY requirements.docker.txt /app/requirements.docker.txt

RUN pip install -r /app/requirements.docker.txt

RUN wget -P /app http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

COPY ./*.py /app/

WORKDIR /app

ENTRYPOINT ["python", "/app/neural_style.py"]
