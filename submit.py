import sys, skvideo.io, json, base64
import numpy as np
import util
from PIL import Image
from io import BytesIO, StringIO

file = sys.argv[-1]
if file == 'demo.py':
  print ("Error loading video")
  quit

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

model = util.create_model()
util.compile_model(model)
model.load_weights("road.h5")

for i, rgb_frame in enumerate(video, start=1):
    red = rgb_frame[:,:,0]
    binary_car_result = np.where(red>250,1,0).astype('uint8')
    road_infer = model.predict(np.array([util.preprocess_input_image(rgb_frame, util.preprocess_opts)]), batch_size=1)[0]
    road_infer = util.postprocess_output(road_infer, util.preprocess_opts)
    binary_road_result = np.zeros_like(binary_car_result)
    binary_road_result[road_infer > 0.5] = 1
    answer_key[i] = [encode(binary_car_result), encode(binary_road_result)]

print (json.dumps(answer_key))
