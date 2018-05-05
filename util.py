import cv2
import numpy as np

def read_image(path):
  """Ensure images read in RGB format (used by moviepy and skvideo)
     Example: f = util.read_image("/tmp/Train/CameraRGB/164.png")"""
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

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
