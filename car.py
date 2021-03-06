import losses
import numpy as np
from keras.layers import Input, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from layers import conv, inception

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

def create_model(opt):
  """Create neural network model, defining layer architecture."""
  dim_y = int((opt['crop_max_y'] - opt['crop_min_y']) / opt['scale_factor'])
  dim_x = int((opt['crop_max_x'] - opt['crop_min_x']) / opt['scale_factor'])

  inputs = Input(shape=(dim_y,dim_x,3))

  conv_1_1 = inception(inputs,128)
  conv_1_2 = inception(conv_1_1,128)

  pool_1 = MaxPooling2D()(conv_1_2)

  conv_2_1 = inception(pool_1,256)
  conv_2_2 = inception(conv_2_1,256)

  pool_2 = MaxPooling2D()(conv_2_2)

  conv_3_1 = inception(pool_2,512)
  conv_3_2 = inception(conv_3_1,512)
  conv_3_3 = inception(conv_3_2,512)

  pool_3 = MaxPooling2D()(conv_3_3)

  conv_4_1 = inception(pool_3,1024)
  conv_4_2 = inception(conv_4_1,1024)
  conv_4_3 = inception(conv_4_2,1024)

  pool_4 = MaxPooling2D()(conv_4_3)

  conv_5_1 = inception(pool_4,1024)
  conv_5_2 = inception(conv_5_1,1024)
  conv_5_3 = inception(conv_5_2,1024)
  conv_5_4 = inception(conv_5_3,1024)
  conv_5_5 = inception(conv_5_4,1024)
  conv_5_6 = inception(conv_5_5,1024)

  unpool_4 = UpSampling2D()(conv_5_6)
  unpool_4 = Concatenate()([unpool_4, conv_4_3]) # skip layer

  decode_4_1 = inception(unpool_4,1024)
  decode_4_2 = inception(decode_4_1,1024)
  decode_4_3 = inception(decode_4_2,512)

  unpool_3 = UpSampling2D()(decode_4_3)
  unpool_3 = Concatenate()([unpool_3, conv_3_3]) # skip layer

  decode_3_1 = inception(unpool_3,512)
  decode_3_2 = inception(decode_3_1,512)
  decode_3_3 = inception(decode_3_2,256)

  unpool_2 = UpSampling2D()(decode_3_3)
  unpool_2 = Concatenate()([unpool_2, conv_2_2]) # skip layer

  decode_2_1 = inception(unpool_2,256)
  decode_2_2 = inception(decode_2_1,128)

  unpool_1 = UpSampling2D()(decode_2_2)
  unpool_1 = Concatenate()([unpool_1, conv_1_2]) # skip layer

  decode_1_1 = inception(unpool_1,128)

  final_layer = Conv2D(1, 3, padding='same', kernel_regularizer=l2(0.01))(decode_1_1)
  final_layer = Activation('sigmoid')(final_layer)
  final_layer = Reshape((dim_y,dim_x))(final_layer)

  model = Model(inputs=inputs, outputs = final_layer, name="Car")

  return model

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer=Adam(amsgrad=True),
                loss=losses.weighted_mean_squared_error(10.0),
                metrics=[losses.f_score(2.0), losses.precision, losses.recall])
