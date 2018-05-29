import cv2
import numpy as np
import random
from time import time
import util

def rgb_shuffle(img,car_mask,road_mask,other_img=None,other_car_mask=None):
  channels = cv2.split(img)
  random.shuffle(channels)
  return (cv2.merge(channels), car_mask, road_mask)

def flip(img,car_mask,road_mask,other_img=None,other_car_mask=None):
  return (np.flip(img,1), np.flip(car_mask,1), np.flip(road_mask,1))

def copy_vehicles(img,car_mask,road_mask,other_img,other_car_mask):
  if other_img is None:
    return (img,car_mask,road_mask)
  # TODO: Add random scaling between 0.75 and 1.00
  # TODO: Add random rotation between -10 and +10 degrees
  roll0 = random.randint(int(-0.05 * img.shape[0]),int(0.05 * img.shape[0]))
  roll1 = random.randint(int(-0.05 * img.shape[1]),int(0.05 * img.shape[1]))
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

class sample_generator(object):

  def __init__(self, examples, category, batch_size=10, augmentations_per_example=0):
    self.batch_size = batch_size
    self.all_aug_labels = []
    self.all_training_input = {}
    self.all_training_output = {}
    augmentation = pick_range_of_augmentations(0, 3)
    print("Reading " + category + " training data...")
    start_time = time()
    examples_for_augmentation = []
    random.shuffle(examples)
    for i in examples:
      img = util.read_train_image(i)
      road_mask, car_mask = util.read_masks(i)
      mask = None
      if category == "car":
        mask = car_mask
      elif category == "road":
        mask = road_mask
      else:
        raise Exception('Unrecognized category: ' + str(category))
      self.all_aug_labels.append(i)
      self.all_training_input[i] = util.preprocess_input_image(img,util.preprocess_opts)
      self.all_training_output[i] = util.preprocess_mask(mask,util.preprocess_opts)
      examples_for_augmentation.append((img,car_mask))
      for j in range(augmentations_per_example):
        aug_label = str(i) + " " + str(j)
        prev_ex = random.choice(examples_for_augmentation)
        aug_img, aug_car_mask, aug_road_mask = augmentation(img,car_mask,road_mask,prev_ex[0],prev_ex[1])
        self.all_aug_labels.append(aug_label)
        self.all_training_input[aug_label] = util.preprocess_input_image(aug_img,util.preprocess_opts)
        if category == "car":
          self.all_training_output[aug_label] = util.preprocess_mask(aug_car_mask,util.preprocess_opts)
        elif category == "road":
          self.all_training_output[aug_label] = util.preprocess_mask(aug_road_mask,util.preprocess_opts)
        else:
          raise Exception('Unrecognized category: ' + str(category))
    elapsed = time() - start_time
    print("    Spent %.0f seconds reading training data." % (elapsed))

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    selected = random.sample(self.all_aug_labels, self.batch_size)
    images = []
    masks = []
    for i in selected:
      images.append(self.all_training_input[i])
      masks.append(self.all_training_output[i])
    return (np.array(images), np.array(masks))
