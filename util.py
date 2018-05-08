import cv2
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2

def read_image(path):
  """Ensure images read in RGB format (used by moviepy and skvideo).
     Example: f = util.read_image("/tmp/Train/CameraRGB/164.png")"""
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def write_image(path,img):
  """Handles RGB or grayscale images."""
  if len(img.shape) == 3 and img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(path, img)

def write_mask(path,mask):
  img = np.zeros_like(mask)
  img[mask > 0] = 254
  write_image(path,img)

def write_probability(path,probability):
  img = np.zeros_like(probability).astype(np.uint8)
  img = np.stack((img,)*3, -1)
  img[probability > 0.2] = [135,60,0] # brown
  img[probability > 0.4] = [254,0,0] # red
  img[probability > 0.5] = [254,135,0] # orange
  img[probability > 0.6] = [254,254,0] # yellow
  img[probability > 0.8] = [254,254,254] # white
  write_image(path,img)

def convert_image_to_road_mask(img):
  """Convert image from Lyft's semantic segmentation format to road mask.
     Lyft's format has category in red channel and 0 in other channels.
     Categories 6 (lane line) and 7 (road) are both considered road for
     this challenge.
     Example: img = util.read_image("/tmp/Train/CameraSeg/164.png")
              m = util.convert_image_to_road_mask(img)"""
  mask = img.max(axis=2)
  road_mask = np.zeros_like(mask)
  road_mask[mask == 6] = 1
  road_mask[mask == 7] = 1
  return road_mask

def read_train_image(example_number):
  """Read image for training example."""
  return read_image("/tmp/Train/CameraRGB/" + str(example_number) + ".png")

def read_masks(example_number):
  """Read road and car masks for training example."""
  img = read_image("/tmp/Train/CameraSeg/" + str(example_number) + ".png")
  road_mask = convert_image_to_road_mask(img)
  car_mask = None # Fix later. Focusing on road for now.
  return road_mask, car_mask

preprocess_opts = {'original_max_x': 800, 'original_max_y': 600,
                   'crop_min_x': 0, 'crop_min_y': 210,
                   'crop_max_x': 800, 'crop_max_y': 526,
                   'scale_factor': 2}
                   # can crop 75 off the bottom and 212 off the top

def crop(img,opt):
  return img[opt['crop_min_y']:opt['crop_max_y'], opt['crop_min_x']:opt['crop_max_x']]

def uncrop_image(img,opt):
  target_shape = (opt['original_max_y'],opt['original_max_x'], img.shape[2])
  frame = np.zeros(target_shape, dtype="uint8")
  frame[opt['crop_min_y']:opt['crop_max_y'], opt['crop_min_x']:opt['crop_max_x'], 0:(img.shape[2])] = img
  img = frame
  return img

def uncrop_output(img,opt):
  target_shape = (opt['original_max_y'],opt['original_max_x'])
  frame = np.zeros(target_shape, dtype="float32")
  frame[opt['crop_min_y']:opt['crop_max_y'], opt['crop_min_x']:opt['crop_max_x']] = img
  img = frame
  return img

def scale(img,opt):
  img = cv2.resize(img, None, fx=(1.0/opt['scale_factor']), fy=(1.0/opt['scale_factor']),
                   interpolation=cv2.INTER_AREA)
  return img

def unscale(img,opt):
  img = cv2.resize(img, None, fx=opt['scale_factor'], fy=opt['scale_factor'])
  return img

def white_balance(img):
  """Ensure that image uses the full brightness range to reduce the effects of lighting."""
  low = np.amin(img)
  high = np.amax(img)
  return (((img - low + 1.0) * 252.0 / (high - low)) - 0.5).astype(np.uint8)

def preprocess_input_image(img,opt):
  """Convert dashcam image to model's input format."""
  img = white_balance(img)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = crop(img,opt)
  img = scale(img,opt)
  return ((img / 253.0) - 0.5).astype(np.float32)

def preprocess_mask(img,opt):
  img = crop(img,opt)
  img = scale(img,opt)
  return img

def postprocess_output(img,opt):
  img = unscale(img,opt)
  img = uncrop_output(img,opt)
  return img

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss='mean_absolute_error',
                metrics=['binary_accuracy', 'binary_crossentropy'])

def create_model():
  """Create neural network model, defining layer architecture."""
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Conv2D(20, (5, 5), padding='same', input_shape=(158,400,3))) # Need to calculate 600->158 instead of hardcode
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
  model.add(Reshape((158,400)))
  compile_model(model)
  return model
