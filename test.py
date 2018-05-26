import numpy as np
import os
import training
import util, car, road

examples = [75,251,270]

os.makedirs("/tmp/output/preprocessing", exist_ok=True)
os.makedirs("/tmp/output/infer_car", exist_ok=True)
os.makedirs("/tmp/output/infer_road", exist_ok=True)
os.makedirs("/tmp/output/augment", exist_ok=True)


car_model = car.create_model(util.preprocess_opts)
car.compile_model(car_model)
car_model.load_weights("car.h5")

road_model = road.create_model(util.preprocess_opts)
road.compile_model(road_model)
road_model.load_weights("road.h5")

def write_augmentation(example, img, car_mask, road_mask, prev_ex, augmentation_name, augmentation):
  base = "/tmp/output/augment/" + str(ex) + "_"
  img, car_mask, road_mask = augmentation(img, car_mask, road_mask, prev_ex[0], prev_ex[1])
  util.write_image(base + augmentation_name + ".png", img)
  util.write_mask(base + augmentation_name + "_car.png", car_mask)
  util.write_mask(base + augmentation_name + "_road.png", road_mask)

prev_ex = (None,None)

for ex in examples:
  pre_out = "/tmp/output/preprocessing/" + str(ex)
  car_out = "/tmp/output/infer_car/" + str(ex)
  road_out = "/tmp/output/infer_road/" + str(ex)
  augment_out = "/tmp/output/augment/" + str(ex)
  img = util.read_train_image(ex)
  util.write_image(pre_out + ".png", img)
  util.write_image(car_out + ".png", img)
  util.write_image(road_out + ".png", img)
  util.write_image(augment_out + ".png", img)
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
  for aug_name in training.all_augmentations.keys():
    write_augmentation(ex, img, car_mask, road_mask, prev_ex, aug_name, training.all_augmentations[aug_name])
  for i in range(1,3):
    write_augmentation(ex, img, car_mask, road_mask, prev_ex, "pick" + str(i), training.pick_n_augmentations(i))
  prev_ex = (img,car_mask)
