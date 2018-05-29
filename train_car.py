import numpy as np
import os.path
import random
import util
import car
from time import time
import training
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = car.create_model(util.preprocess_opts)
car.compile_model(model)
if os.path.exists("car.h5"):
  print("Loading existing model from car.h5")
  model.load_weights("car.h5")

validation_size = 350
(trn,val) = util.validation_split(util.all_examples(),validation_size)

batch_size = 20
train_generator = training.sample_generator(category="car",
                                            examples=trn,
                                            batch_size=batch_size,
                                            augmentations_per_example=5)
validation_generator = training.sample_generator(category="car",
                                                 examples=val,
                                                 batch_size=batch_size)

stop_early = EarlyStopping(monitor='val_fscore', patience=50, mode='max', verbose=1)
save_best = ModelCheckpoint(filepath='car.h5', monitor='val_fscore', mode='max',
                            save_best_only=True, save_weights_only=True, verbose=1)

model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=1000/batch_size,
                    validation_steps=int(2*validation_size/batch_size),
                    callbacks=[stop_early, save_best],
                    epochs=200)
