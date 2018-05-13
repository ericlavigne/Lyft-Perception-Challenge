import sys, skvideo.io, json, base64
import numpy as np
import car, road, util
from PIL import Image
from io import BytesIO, StringIO
from time import time

file = sys.argv[-1]
if file == 'demo.py':
  print("Error loading video")
  quit

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

car_model = car.create_model(util.preprocess_opts)
road_model = road.create_model(util.preprocess_opts)
util.compile_model(car_model)
util.compile_model(road_model)
car_model.load_weights("car.h5")
road_model.load_weights("road.h5")

steps = ['preprocess','car_predict','road_predict','car_postprocess','road_postprocess','encode_array','encode_png','encode_json']
prof = {}
for s in steps:
    prof[s] = 0.0
total_frames = 0

for i, rgb_frame in enumerate(video, start=1):
    red = rgb_frame[:,:,0]
    start_time = time()

    preprocessed = util.preprocess_input_image(rgb_frame, util.preprocess_opts)
    preprocess_time = time()
    prof['preprocess'] += (preprocess_time - start_time)

    car_infer = car_model.predict(np.array([preprocessed]), batch_size=1)[0]
    car_predict_time = time()
    prof['car_predict'] += (car_predict_time - preprocess_time)

    car_infer = util.postprocess_output(car_infer, util.preprocess_opts)
    car_postprocess_time = time()
    prof['car_postprocess'] += (car_postprocess_time - car_predict_time)

    road_infer = road_model.predict(np.array([preprocessed]), batch_size=1)[0]
    road_predict_time = time()
    prof['road_predict'] += (road_predict_time - car_postprocess_time)

    road_infer = util.postprocess_output(road_infer, util.preprocess_opts)
    road_postprocess_time = time()
    prof['road_postprocess'] += (road_postprocess_time - road_predict_time)

    binary_road_result = np.where(road_infer > 0.5,1,0).astype('uint8')
    binary_car_result = np.where(car_infer > 0.5,1,0).astype('uint8')
    binary_road_result[binary_car_result == 1] = 0 # if uncertain, safer to assume car than road
    encode_array_time = time()
    prof['encode_array'] += (encode_array_time - road_postprocess_time)

    answer_key[i] = [encode(binary_car_result), encode(binary_road_result)]
    encode_png_time = time()
    prof['encode_png'] += (encode_png_time - encode_array_time)

    total_frames = total_frames + 1

before_json = time()
json_result = json.dumps(answer_key)
after_json = time()

sys.stderr.write('\nTimings across %d frames:\n' % (total_frames))
for s in steps:
    sys.stderr.write('    %s :   %.0f   (%.3f / frame)\n' % (s, prof[s], prof[s] / total_frames))
sys.stderr.write('\n')

print(json_result)
