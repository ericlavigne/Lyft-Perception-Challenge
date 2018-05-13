import cv2
import numpy as np
import random

def rgb_shuffle(img,car_mask,road_mask):
  channels = cv2.split(img)
  random.shuffle(channels)
  return (cv2.merge(channels), car_mask, road_mask)

def flip(img,car_mask,road_mask):
  return (np.flip(img,1), np.flip(car_mask,1), np.flip(road_mask,1))

all_augmentations = {'rgb_shuffle': rgb_shuffle, 'flip': flip}

def pick_n_augmentations(qty):
  def pick_exactly_aux(img,car_mask,road_mask):
    selected = random.sample(list(all_augmentations.values()), qty)
    for x in selected:
      img,car_mask,road_mask = x(img,car_mask,road_mask)
    return (img,car_mask,road_mask)
  return pick_exactly_aux

def pick_range_of_augmentations(min_qty, max_qty):
  options = range(min_qty, max_qty + 1)
  def pick_range_aux(img,car_mask,road_mask):
    return pick_n_augmentations(random.sample(options,1)[0])(img,car_mask,road_mask)
  return pick_range_aux
