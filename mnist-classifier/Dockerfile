FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y \
  python-pip \
  python-dev \
  python-virtualenv 
RUN virtualenv --system-site-packages ~/tensorflow 
RUN /bin/bash -c "source ~/tensorflow/bin/activate"
RUN pip install --upgrade tensorflow 

COPY mnist_softmax.py .

CMD python mnist_softmax.py