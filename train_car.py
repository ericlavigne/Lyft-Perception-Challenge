import numpy as np
import os.path
import random
import util
import car

class sample_generator(object):

  def __init__(self, batch_size=10):
    self.batch_size = batch_size
    self.all_examples = list(range(1000))

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    selected = random.sample(self.all_examples, self.batch_size)
    images = []
    masks = []
    for i in selected:
      images.append(util.preprocess_input_image(util.read_train_image(i),util.preprocess_opts))
      road_mask, car_mask = util.read_masks(i)
      masks.append(util.preprocess_mask(car_mask,util.preprocess_opts))
    return (np.array(images), np.array(masks))

model = car.create_model(util.preprocess_opts)
util.compile_model(model)
if os.path.exists("car.h5"):
  print("Loading existing model from car.h5")
  model.load_weights("car.h5")
model.fit_generator(sample_generator(batch_size=10),
                    steps_per_epoch=20,
                    epochs=50)
model.save_weights("car.h5")
