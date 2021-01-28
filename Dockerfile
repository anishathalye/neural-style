FROM python:3.7-buster

RUN apt-get update

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

RUN wget -P /app http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

COPY ./*.py /app/

WORKDIR /app

ENTRYPOINT ["python", "/app/neural_style.py"]
