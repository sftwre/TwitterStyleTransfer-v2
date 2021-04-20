
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

WORKDIR /root

RUN apt-get update && apt-get install python3.7

RUN pip install --upgrade pip

RUN mkdir ./runs

RUN mkdir ./models

# Install library dependencies
RUN pip install -r requirements.txt

# copy data
COPY data/tweets.train.txt ./data/tweets.train.txt
COPY data/tweets.train.labels ./data/tweets.train.labels
COPY data/tweets.test.txt ./data/tweets.test.txt
COPY data/tweets.test.labels ./data/tweets.test.labels
COPY data/vocab.txt ./data/vocab.txt

# copy trainer package
COPY trainer/* ./trainer/


