#!/bin/bash

pip install scikit-video

cd /tmp; wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz

cd /tmp; tar -xzf lyft_training_data.tar.gz
