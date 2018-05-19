import cv2
from glob import glob
import numpy as np
import tensorflow as tf
import car
import road
from keras import backend as K

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

def all_examples():
  return [x.split('/')[-1].split('.')[0] for x in glob("/tmp/Train/CameraSeg/*.png")]

def read_train_image(example):
  """Read image for training example."""
  return read_image("/tmp/Train/CameraRGB/" + str(example) + ".png")

def read_masks(example):
  """Read road and car masks for training example."""
  img = read_image("/tmp/Train/CameraSeg/" + str(example) + ".png")
  road_mask = road.convert_image_to_mask(img)
  car_mask = car.convert_image_to_mask(img)
  return road_mask, car_mask

preprocess_opts = {'original_max_x': 800, 'original_max_y': 600,
                   'crop_min_x': 0, 'crop_min_y': 210,
                   'crop_max_x': 800, 'crop_max_y': 530,
                   'scale_factor': 5}
                   # can crop 75 off the bottom and 212 off the top
                   # Need cropped and scaled x/y dimensions to be
                   # divisible by 16 to allow pooling/unpooling

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

def mask_percentage(img):
  return len(np.where(img == 1)[0]) * 100.0 / len(img.flatten())
