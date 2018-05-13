import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
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
  model.add(Conv2D(20, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(20, (5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Conv2D(1, (5, 5), padding='same', kernel_regularizer=l2(0.01)))
  model.add(Activation('sigmoid'))
  model.add(Reshape((dim_y,dim_x)))
  return model

def weighted_binary_crossentropy(weight):
  """Higher weights increase the importance of examples in which
     the correct answer is 1. Higher values should be used when
     1 is a rare answer. Lower values should be used when 0 is
     a rare answer."""
  return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss=weighted_binary_crossentropy(20.),
                metrics=['binary_accuracy', 'binary_crossentropy'])
