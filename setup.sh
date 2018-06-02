#!/bin/bash

pip install scikit-video
sudo apt-get update
sudo apt-get install -y cuda-libraries-9-0
pip install tensorflow-gpu==1.8
pip install Keras==2.1.6
pip install pyzmq

cd /tmp

# Udacity provided the first CARLA data set as part of the competition announcement.
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz
tar -xzf lyft_training_data.tar.gz

# Ong Chin-Kiat (chinkiat) donated some extra CARLA data sets.
wget https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip
unzip carla-capture-20180513A.zip -d Train
wget https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180528.zip
unzip carla-capture-20180528.zip -d Train

# Notes on non-automated parts:

# Phu Nguyen (phmagic) donated this dataset https://www.dropbox.com/s/1etgf32uye2iy8q/world_2_w_cars.tar.gz?dl=1
# It's tricky to include due to overlapping filenames in different folders. I will add prefixes so it will fit
# in my current file structure. gzip claims that this file is not in gzip format, so I unzipped on MacOS desktop
# where Archive Utility autodetects the format then zipped and transferred via scp.
# scp /Users/ericlavigne/Downloads/world_2_little_cars.zip eric@192.168.1.97:/tmp/phmagic-2018-05-28.zip
# cd /tmp/world_2_little_cars
# for d in *; do cd $d ; cd CameraRGB; for f in *.png; do cp -- "$f" "/tmp/Train/CameraRGB/phmagic-$d-$f"; done; cd ../..; done
# for d in *; do cd $d ; cd CameraSeg; for f in *.png; do cp -- "$f" "/tmp/Train/CameraSeg/phmagic-$d-$f"; done; cd ../..; done

# Mohamed Eltohamy donated this dataset https://drive.google.com/open?id=1NimO26IH_Y8DziDMsgBCZeHlT3duj4-e
# Download from Google Drive is difficult to automate, so I downloaded on desktop and transferred.
# scp /Users/ericlavigne/Downloads/Carla_Town2_1000_images-20180602T175805Z-001.zip eric@192.168.1.97:/tmp/Mohamed-2018-06-02.zip
# unzip Mohamed-2018-06-02.zip
# cp Carla_Town2_1000_images/CameraRGB/* Train/CameraRGB/
# cp Carla_Town2_1000_images/SegCamera/* Train/CameraSeg/
# Mohamed's dataset has C_T2_E_1_ prefix.
