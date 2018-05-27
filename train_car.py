import numpy as np
import os.path
import random
import util
import car
from time import time
import training
from keras.callbacks import EarlyStopping, ModelCheckpoint

class sample_generator(object):

  def __init__(self, examples, batch_size=10, augmentations_per_example=0):
    self.batch_size = batch_size
    self.all_aug_labels = []
    self.all_training_input = {}
    self.all_training_output = {}
    augmentation = training.pick_range_of_augmentations(0, 3)
    print("Reading car training data...")
    start_time = time()
    examples_for_augmentation = []
    random.shuffle(examples)
    for i in examples:
      img = util.read_train_image(i)
      road_mask, car_mask = util.read_masks(i)
      self.all_aug_labels.append(i)
      self.all_training_input[i] = util.preprocess_input_image(img,util.preprocess_opts)
      self.all_training_output[i] = util.preprocess_mask(car_mask,util.preprocess_opts)
      examples_for_augmentation.append((img,car_mask))
      for j in range(augmentations_per_example):
        aug_label = str(i) + " " + str(j)
        prev_ex = random.choice(examples_for_augmentation)
        aug_img, aug_car_mask, aug_road_mask = augmentation(img,car_mask,road_mask,prev_ex[0],prev_ex[1])
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

validation_size = 200
(trn,val) = util.validation_split(util.all_examples(),validation_size)

batch_size = 100
train_generator = sample_generator(examples=trn,
                                   batch_size=batch_size,
                                   augmentations_per_example=5)
validation_generator = sample_generator(examples=val,
                                        batch_size=batch_size)

stop_early = EarlyStopping(monitor='val_fscore', patience=20, mode='max', verbose=1)
save_best = ModelCheckpoint(filepath='car.h5', monitor='val_fscore', mode='max',
                            save_best_only=True, save_weights_only=True, verbose=1)

model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=1000/batch_size,
                    validation_steps=int(2*validation_size/batch_size),
                    callbacks=[stop_early, save_best],
                    epochs=200)
