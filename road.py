import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2

def convert_image_to_mask(img):
  """Convert image from Lyft's semantic segmentation format to road mask.
     Lyft's format has category in red channel and 0 in other channels.
     Categories 6 (lane line) and 7 (road) are both considered road for
     this challenge.
     Example: img = util.read_image("/tmp/Train/CameraSeg/164.png")
              m = road.convert_image_to_mask(img)"""
  mask = img.max(axis=2)
  road_mask = np.zeros_like(mask)
  road_mask[mask == 6] = 1
  road_mask[mask == 7] = 1
  return road_mask

def create_model(opt):
  """Create neural network model, defining layer architecture."""
  dim_y = int((opt['crop_max_y'] - opt['crop_min_y']) / opt['scale_factor'])
  dim_x = int((opt['crop_max_x'] - opt['crop_min_x']) / opt['scale_factor'])
  model = Sequential()
  model.add(Conv2D(20, (5, 5), padding='same', input_shape=(dim_y,dim_x,3)))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same', dilation_rate=2))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same', dilation_rate=2))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same', dilation_rate=3))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same', dilation_rate=4))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(1, (5, 5), padding='same', dilation_rate=4, kernel_regularizer=l2(0.01)))
  model.add(Activation('sigmoid'))
  model.add(Reshape((dim_y,dim_x)))
  return model
