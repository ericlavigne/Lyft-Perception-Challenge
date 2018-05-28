#!/bin/bash

pip install --upgrade pip
pip install scikit-video
sudo apt-get update
sudo apt-get install -y cuda-libraries-9-0
pip install tensorflow-gpu==1.8
pip install Keras==2.1.6

cd /tmp

wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz
tar -xzf lyft_training_data.tar.gz

wget https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip
unzip carla-capture-20180513A.zip -d Train

wget https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180528.zip
unzip carla-capture-20180528.zip -d Train
