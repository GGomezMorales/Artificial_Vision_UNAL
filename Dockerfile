FROM ubuntu:22.04

# Install the necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip 

RUN pip3 install numpy \
    pandas \
    openCV-python \
    matplotlib \
    jupyter \
    ipykernel

RUN pip install notebook

WORKDIR /home/artificial-vision

# COPY /python-scripts-notebooks /home/artificial-vision/


