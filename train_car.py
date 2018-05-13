import numpy as np
import os.path
import random
import util
import car
from time import time

class sample_generator(object):

  def __init__(self, batch_size=10):
    self.batch_size = batch_size
    self.all_examples = list(range(1000))
    self.all_training_input = []
    self.all_training_output = []
    print("Reading car training data...")
    start_time = time()
    for i in range(1000):
      self.all_training_input.append(util.preprocess_input_image(util.read_train_image(i),util.preprocess_opts))
      road_mask, car_mask = util.read_masks(i)
      self.all_training_output.append(util.preprocess_mask(car_mask,util.preprocess_opts))
    elapsed = time() - start_time
    print("    Spent %.0f seconds reading training data." % (elapsed))

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    selected = random.sample(self.all_examples, self.batch_size)
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
model.fit_generator(sample_generator(batch_size=batch_size),
                    steps_per_epoch=1000/batch_size,
                    epochs=30)

model.save_weights("car.h5")
