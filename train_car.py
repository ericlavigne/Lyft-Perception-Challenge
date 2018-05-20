import numpy as np
import os.path
import random
import util
import car
from time import time
import training

class sample_generator(object):

  def __init__(self, batch_size=10, max_examples=None, augmentations_per_example=1):
    self.batch_size = batch_size
    self.all_aug_labels = []
    self.all_training_input = {}
    self.all_training_output = {}
    augmentation = training.pick_range_of_augmentations(0, 2)
    example_labels = util.all_examples()
    if max_examples is not None:
      example_labels = random.sample(example_labels, max_examples)
    print("Reading car training data...")
    start_time = time()
    for i in example_labels:
      img = util.read_train_image(i)
      road_mask, car_mask = util.read_masks(i)
      for j in range(augmentations_per_example):
        aug_label = str(i) + " " + str(j)
        aug_img, aug_car_mask, aug_road_mask = augmentation(img,car_mask,road_mask)
        self.all_aug_labels.append(aug_label)
        self.all_training_input[aug_label] = util.preprocess_input_image(aug_img,util.preprocess_opts)
        self.all_training_output[aug_label] = util.preprocess_mask(aug_car_mask,util.preprocess_opts)
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

model = car.create_model(util.preprocess_opts)
car.compile_model(model)
if os.path.exists("car.h5"):
  print("Loading existing model from car.h5")
  model.load_weights("car.h5")

batch_size = 10
train_generator = sample_generator(batch_size=batch_size,
                                   augmentations_per_example=5)
validation_generator = sample_generator(batch_size=batch_size,
                                        max_examples=100,
                                        augmentations_per_example=5)
num_savepoints = 3
for i in range(num_savepoints):
  print("Training batch " + str(i+1) + " of " + str(num_savepoints))
  model.fit_generator(train_generator,
                      validation_data=validation_generator,
                      steps_per_epoch=1000/batch_size,
                      validation_steps=300/batch_size,
                      epochs=30)
  model.save_weights("car.h5")
