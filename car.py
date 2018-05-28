import losses
import numpy as np
from keras.layers import Input, Concatenate
from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
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
  layer3 = Activation('elu')(layer2)
  return layer3

def inception(prev, depth):
  """Based on second InceptionV3 module whose output depth is 288.
     Scale all component depths based on ratio between desired depth
     and 288. Prevent any component depth from dropping below 16."""
  scale48 = max(16, int(48 * depth / 288))
  scale64 = max(16, int(64 * depth / 288))
  scale96 = max(16, int(96 * depth / 288))

  layer_1 = conv(prev, scale64, 1)

  layer_2_1 = conv(prev, scale48, 1)
  layer_2_2 = conv(layer_2_1, scale64, 5)

  layer_3_1 = conv(prev, scale64, 1)
  layer_3_2 = conv(layer_3_1, scale96, 3)
  layer_3_3 = conv(layer_3_2, scale96, 3)

  layer_4_1 = AveragePooling2D(pool_size=3, strides=1, padding='same')(prev)
  layer_4_2 = conv(layer_4_1, scale64, 1)

  return Concatenate()([layer_1, layer_2_2, layer_3_3, layer_4_2])

def create_model(opt):
  """Create neural network model, defining layer architecture."""
  dim_y = int((opt['crop_max_y'] - opt['crop_min_y']) / opt['scale_factor'])
  dim_x = int((opt['crop_max_x'] - opt['crop_min_x']) / opt['scale_factor'])

  inputs = Input(shape=(dim_y,dim_x,3))

  conv_1_1 = inception(inputs,64)
  conv_1_2 = inception(conv_1_1,64)

  pool_1 = MaxPooling2D()(conv_1_2)

  conv_2_1 = inception(pool_1,128)
  conv_2_2 = inception(conv_2_1,128)

  pool_2 = MaxPooling2D()(conv_2_2)

  conv_3_1 = inception(pool_2,256)
  conv_3_2 = inception(conv_3_1,256)
  conv_3_3 = inception(conv_3_2,256)

  pool_3 = MaxPooling2D()(conv_3_3)

  conv_4_1 = inception(pool_3,512)
  conv_4_2 = inception(conv_4_1,512)
  conv_4_3 = inception(conv_4_2,512)

  pool_4 = MaxPooling2D()(conv_4_3)

  conv_5_1 = inception(pool_4,512)
  conv_5_2 = inception(conv_5_1,512)
  conv_5_3 = inception(conv_5_2,512)
  conv_5_4 = inception(conv_5_3,512)
  conv_5_5 = inception(conv_5_4,512)
  conv_5_6 = inception(conv_5_5,512)

  unpool_4 = UpSampling2D()(conv_5_6)
  unpool_4 = Concatenate()([unpool_4, conv_4_3]) # skip layer

  decode_4_1 = inception(unpool_4,512)
  decode_4_2 = inception(decode_4_1,512)
  decode_4_3 = inception(decode_4_2,256)

  unpool_3 = UpSampling2D()(decode_4_3)
  unpool_3 = Concatenate()([unpool_3, conv_3_3]) # skip layer

  decode_3_1 = inception(unpool_3,256)
  decode_3_2 = inception(decode_3_1,256)
  decode_3_3 = inception(decode_3_2,128)

  unpool_2 = UpSampling2D()(decode_3_3)
  unpool_2 = Concatenate()([unpool_2, conv_2_2]) # skip layer ######

  decode_2_1 = inception(unpool_2,128)
  decode_2_2 = inception(decode_2_1,64)

  unpool_1 = UpSampling2D()(decode_2_2)
  unpool_1 = Concatenate()([unpool_1, conv_1_2]) # skip layer

  decode_1_1 = inception(unpool_1,64)

  final_layer = Conv2D(1, 3, padding='same', kernel_regularizer=l2(0.01))(decode_1_1)
  final_layer = Activation('sigmoid')(final_layer)
  final_layer = Reshape((dim_y,dim_x))(final_layer)

  model = Model(inputs=inputs, outputs = final_layer, name="Car")

  return model

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer=Adam(amsgrad=True),
                loss=losses.balanced_binary_mean_squared_error, # losses.f_score_loss(2.0),
                metrics=[losses.f_score(2.0), losses.precision, losses.recall])
