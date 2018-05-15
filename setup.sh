#!/bin/bash

pip install scikit-video
cd /tmp
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz
tar -xzf lyft_training_data.tar.gz
wget https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip
unzip carla-capture-20180513A.zip -d Train
