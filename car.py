import losses
import numpy as np
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

def convert_image_to_mask(img):
  """Convert image from Lyft's semantic segmentation format to car mask.
     Lyft's format has category in red channel and 0 in other channels.
     Category 10 is for vehicles, but includes the hood of the main car
     which should not be classified as a vehicle for this challenge.
     Example: img = util.read_image("/tmp/Train/CameraSeg/164.png")
              m = car.convert_image_to_mask(img)"""
  mask = img.max(axis=2)
  car_mask = np.zeros_like(mask)
  # Extract vehicle label which includes the hood
  car_mask[mask == 10] = 1
  # Everything below y=517 is part of the hood
  car_mask[517:,:] = 0
  # The center of the hood reaches y=497 between x values of 108 and 691
  car_mask[497:,108:691] = 0
  return car_mask

def conv(previous_layer, depth, kernel):
  layer1 = Conv2D(depth, kernel, padding='same')(previous_layer)
  layer2 = BatchNormalization()(layer1)
  layer3 = Activation('tanh')(layer2)
  return layer3

def create_model(opt):
  """Create neural network model, defining layer architecture."""
  dim_y = int((opt['crop_max_y'] - opt['crop_min_y']) / opt['scale_factor'])
  dim_x = int((opt['crop_max_x'] - opt['crop_min_x']) / opt['scale_factor'])

  inputs = Input(shape=(dim_y,dim_x,3))

  conv_1_1 = conv(inputs,64,3)
  conv_1_2 = conv(conv_1_1,64,3)

  pool_1 = MaxPooling2D()(conv_1_2)

  conv_2_1 = conv(pool_1,128,3)
  conv_2_2 = conv(conv_2_1,128,3)

  pool_2 = MaxPooling2D()(conv_2_2)

  conv_3_1 = conv(pool_2,256,3)
  conv_3_2 = conv(conv_3_1,256,3)
  conv_3_3 = conv(conv_3_2,256,3)
  conv_3_4 = conv(conv_3_3,256,3)

  unpool_2 = UpSampling2D()(conv_3_4)

  conv_4_1 = conv(unpool_2,128,3)
  conv_4_2 = conv(conv_4_1,128,3)

  unpool_1 = UpSampling2D()(conv_4_2)


  conv_5_1 = conv(unpool_1,64,3)
  conv_5_2 = conv(conv_5_1,64,3)

  conv_6 = Conv2D(1, (3, 3), padding='same', kernel_regularizer=l2(0.01))(conv_5_2)
  conv_6 = Activation('sigmoid')(conv_6)
  conv_6 = Reshape((dim_y,dim_x))(conv_6)

  model = Model(inputs=inputs, outputs = conv_6, name="Car")

  return model

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss=losses.balanced_binary_mean_squared_error,
                metrics=['binary_accuracy', losses.weighted_mean_squared_error(34.6)])
