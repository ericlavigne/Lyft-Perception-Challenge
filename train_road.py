import numpy as np
import os.path
import random
import util
import road

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
      masks.append(util.preprocess_mask(road_mask,util.preprocess_opts))
    return (np.array(images), np.array(masks))

model = road.create_model(util.preprocess_opts)
road.compile_model(model)
if os.path.exists("road.h5"):
  print("Loading existing model from road.h5")
  model.load_weights("road.h5")

batch_size = 10
model.fit_generator(sample_generator(batch_size=batch_size),
                    steps_per_epoch=1000/batch_size,
                    epochs=10)

model.save_weights("road.h5")