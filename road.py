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

  inputs = Input(shape=(dim_y,dim_x,3))

  conv_1_1 = inception(inputs,64)
  conv_1_2 = inception(conv_1_1,64)

  pool_1 = MaxPooling2D()(conv_1_2)

  conv_2_1 = inception(pool_1,128)
  conv_2_2 = inception(conv_2_1,128)

  pool_2 = MaxPooling2D()(conv_2_2)

  conv_3_1 = inception(pool_2,128)
  conv_3_2 = inception(conv_3_1,128)
  conv_3_3 = inception(conv_3_2,128)

  pool_3 = MaxPooling2D()(conv_3_3)

  conv_4_1 = inception(pool_3,256)
  conv_4_2 = inception(conv_4_1,256)
  conv_4_3 = inception(conv_4_2,256)

  pool_4 = MaxPooling2D()(conv_4_3)

  conv_5_1 = inception(pool_4,256)
  conv_5_2 = inception(conv_5_1,256)
  conv_5_3 = inception(conv_5_2,256)
  conv_5_4 = inception(conv_5_3,256)
  conv_5_5 = inception(conv_5_4,256)
  conv_5_6 = inception(conv_5_5,256)

  unpool_4 = UpSampling2D()(conv_5_6)
  unpool_4 = Concatenate()([unpool_4, conv_4_3]) # skip layer

  decode_4_1 = inception(unpool_4,256)
  decode_4_2 = inception(decode_4_1,256)
  decode_4_3 = inception(decode_4_2,128)

  unpool_3 = UpSampling2D()(decode_4_3)
  unpool_3 = Concatenate()([unpool_3, conv_3_3]) # skip layer

  decode_3_1 = inception(unpool_3,128)
  decode_3_2 = inception(decode_3_1,128)
  decode_3_3 = inception(decode_3_2,128)

  unpool_2 = UpSampling2D()(decode_3_3)
  unpool_2 = Concatenate()([unpool_2, conv_2_2]) # skip layer

  decode_2_1 = inception(unpool_2,128)
  decode_2_2 = inception(decode_2_1,64)

  unpool_1 = UpSampling2D()(decode_2_2)
  unpool_1 = Concatenate()([unpool_1, conv_1_2]) # skip layer

  decode_1_1 = inception(unpool_1,64)

  final_layer = Conv2D(1, 3, padding='same', kernel_regularizer=l2(0.01))(decode_1_1)
  final_layer = Activation('sigmoid')(final_layer)
  final_layer = Reshape((dim_y,dim_x))(final_layer)

  model = Model(inputs=inputs, outputs = final_layer, name="Road")

  return model

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer=Adam(amsgrad=True),
                loss=losses.balanced_binary_mean_squared_error, # losses.f_score_loss(0.5),
                metrics=[losses.f_score(0.5), losses.precision, losses.recall])
