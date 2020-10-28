#!/bin/bash

# Get packages required for OpenCV

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

# Need to get an older version of OpenCV because version 4 has errors
pip3 install opencv-python==3.4.6.27
pip3 install imutils
pip3 install scikit-image

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
# Will change to just 'pip3 install tensorflow' once newer versions of TF are added to piwheels

#pip3 install tensorflow

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1TO0l6zH080WSFU50GJpeMjWhpaofdljn" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1TO0l6zH080WSFU50GJpeMjWhpaofdljn" -o tflite_runtime-2.3.0-py3-none-linux_armv7l.whl

echo Download finished.

pip3 install tflite_runtime-2.3.0-py3-none-linux_armv7l.whl

echo Required Installations Finished