import cv2
import numpy as np

def read_image(path):
  """Ensure images read in RGB format (used by moviepy and skvideo)
     Example: f = util.read_image("/tmp/Train/CameraRGB/164.png")"""
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def write_image(path,img):
  """Handles RGB or grayscale images"""
  if len(img.shape) == 3 and img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(path, img)

def write_mask(path,mask):
  img = np.zeros_like(mask)
  img[mask > 0] = 254
  write_image(path,img)

def convert_image_to_road_mask(img):
  """Convert image from Lyft's semantic segmentation format to road mask.
     Lyft's format has category in red channel and 0 in other channels.
     Categories 6 (lane line) and 7 (road) are both considered road for
     this challenge.
     Example: img = util.read_image("/tmp/Train/CameraSeg/164.png")
              m = util.convert_image_to_road_mask(img)"""
  mask = img.max(axis=2)
  road = np.zeros_like(mask)
  road[mask == 6] = 1
  road[mask == 7] = 1
  return road

def read_train_image(example_number):
  """Read image for training example"""
  return read_image("/tmp/Train/CameraRGB/" + str(example_number) + ".png")

def read_masks(example_number):
  """Read road and car masks for training example"""
  img = read_image("/tmp/Train/CameraSeg/" + str(example_number) + ".png")
  road_mask = convert_image_to_road_mask(img)
  car_mask = None # Fix later. Focusing on road for now.
  return road_mask, car_mask
