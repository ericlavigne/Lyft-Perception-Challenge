import cv2
import numpy as np
import random

def rgb_shuffle(img,car_mask,road_mask,other_img=None,other_car_mask=None):
  channels = cv2.split(img)
  random.shuffle(channels)
  return (cv2.merge(channels), car_mask, road_mask)

def flip(img,car_mask,road_mask,other_img=None,other_car_mask=None):
  return (np.flip(img,1), np.flip(car_mask,1), np.flip(road_mask,1))

def copy_vehicles(img,car_mask,road_mask,other_img,other_car_mask):
  if other_img is None:
    return (img,car_mask,road_mask)
  # TODO: Add random scaling between 0.5 and 1.25
  # TODO: Add random rotation between -15 and +15 degrees
  roll0 = random.randint(0,img.shape[0])
  roll1 = random.randint(0,img.shape[1])
  other_img = np.roll(other_img,(roll0,roll1),axis=(0,1))
  other_car_mask = np.roll(other_car_mask,(roll0,roll1),axis=(0,1))
  result_img = np.zeros_like(img)
  result_car_mask = np.copy(car_mask)
  result_car_mask[other_car_mask > 0] = 1
  result_road_mask = np.copy(road_mask)
  result_road_mask[other_car_mask > 0] = 0
  result_img = np.copy(img)
  result_img[other_car_mask > 0] = other_img[other_car_mask > 0]
  return (result_img,result_car_mask,result_road_mask)

all_augmentations = {'rgb_shuffle': rgb_shuffle, 'flip': flip, 'copy_vehicles': copy_vehicles}

def pick_n_augmentations(qty):
  def pick_exactly_aux(img,car_mask,road_mask,other_img,other_car_mask):
    selected = random.sample(list(all_augmentations.values()), qty)
    for x in selected:
      img,car_mask,road_mask = x(img,car_mask,road_mask,other_img,other_car_mask)
    return (img,car_mask,road_mask)
  return pick_exactly_aux

def pick_range_of_augmentations(min_qty, max_qty):
  options = range(min_qty, max_qty + 1)
  def pick_range_aux(img,car_mask,road_mask,other_img,other_car_mask):
    return pick_n_augmentations(random.sample(options,1)[0])(img,car_mask,road_mask,other_img,other_car_mask)
  return pick_range_aux
