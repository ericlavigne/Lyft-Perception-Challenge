import numpy as np
import os
import util

examples = [75]

os.makedirs("/tmp/output/infer_road", exist_ok=True)

model = util.create_model()
util.compile_model(model)
model.load_weights("road.h5")

for ex in examples:
  base_out = "/tmp/output/infer_road/" + str(ex)
  img = util.read_train_image(ex)
  util.write_image(base_out + ".png", img)
  road_mask, car_mask = util.read_masks(ex)
  util.write_mask(base_out + "_truth.png",road_mask)
  road_infer = model.predict(np.array([util.preprocess_input_image(img)]), batch_size=1)[0]
  util.write_probability(base_out + "_infer.png", road_infer)
