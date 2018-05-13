import numpy as np
import os
import util, car, road

examples = [75,251,270]

os.makedirs("/tmp/output/preprocessing", exist_ok=True)
os.makedirs("/tmp/output/infer_car", exist_ok=True)
os.makedirs("/tmp/output/infer_road", exist_ok=True)

car_model = car.create_model(util.preprocess_opts)
util.compile_model(car_model)
car_model.load_weights("car.h5")

road_model = road.create_model(util.preprocess_opts)
util.compile_model(road_model)
road_model.load_weights("road.h5")

for ex in examples:
  pre_out = "/tmp/output/preprocessing/" + str(ex)
  car_out = "/tmp/output/infer_car/" + str(ex)
  road_out = "/tmp/output/infer_road/" + str(ex)
  img = util.read_train_image(ex)
  util.write_image(pre_out + ".png", img)
  util.write_image(car_out + ".png", img)
  util.write_image(road_out + ".png", img)
  road_mask, car_mask = util.read_masks(ex)
  util.write_mask(car_out + "_truth.png",car_mask)
  util.write_mask(road_out + "_truth.png",road_mask)
  cropped = util.crop(img,util.preprocess_opts)
  util.write_image(pre_out + "_crop.png", cropped)
  uncropped = util.uncrop_image(cropped,util.preprocess_opts)
  util.write_image(pre_out + "_uncrop.png", uncropped)
  preprocessed = util.preprocess_input_image(img,util.preprocess_opts)
  car_infer = car_model.predict(np.array([preprocessed]), batch_size=1)[0]
  car_infer = util.postprocess_output(car_infer, util.preprocess_opts)
  util.write_probability(car_out + "_infer.png", car_infer)
  road_infer = road_model.predict(np.array([preprocessed]), batch_size=1)[0]
  road_infer = util.postprocess_output(road_infer, util.preprocess_opts)
  util.write_probability(road_out + "_infer.png", road_infer)
