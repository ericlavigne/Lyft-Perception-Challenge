import numpy as np
import os.path
import random
import util
import road
from time import time
import training
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = road.create_model(util.preprocess_opts)
road.compile_model(model)
if os.path.exists("road.h5"):
  print("Loading existing model from road.h5")
  model.load_weights("road.h5")

validation_size = 350
(trn,val) = util.validation_split(util.all_examples(),validation_size)

batch_size = 20
train_generator = training.sample_generator(category="road",
                                            examples=trn,
                                            batch_size=batch_size,
                                            augmentations_per_example=5)
validation_generator = training.sample_generator(category="road",
                                                 examples=val,
                                                 batch_size=batch_size)

stop_early = EarlyStopping(monitor='val_fscore', patience=50, mode='max', verbose=1)
save_best = ModelCheckpoint(filepath='road.h5', monitor='val_fscore', mode='max',
                            save_best_only=True, save_weights_only=True, verbose=1)

model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=1000/batch_size,
                    validation_steps=int(2*validation_size/batch_size),
                    callbacks=[stop_early, save_best],
                    epochs=200)
