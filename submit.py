import cv2
import sys, skvideo.io, json, base64
from multiprocessing import Process, Pipe
import numpy as np
import random
import car, road, util
from io import BytesIO, StringIO
from time import time, sleep

def encode(array):
  retval, buff = cv2.imencode('.png',array)
  return base64.b64encode(buff).decode("utf-8")

def preprocessor(filename,outpipe):
  try:
    start_time = time()
    video = skvideo.io.vread(file)
    total_frames = 0
    for i, rgb_frame in enumerate(video, start=1):
      preprocessed = util.preprocess_input_image(rgb_frame, util.preprocess_opts)
      outpipe.send([i,preprocessed])
      total_frames = total_frames + 1
      #sys.stderr.write("preproc " + str(i) + "\n")
    preprocess_time = time() - start_time
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("preprocessor", preprocess_time, preprocess_time / total_frames))
    outpipe.send("done")
    outpipe.close()

  except Exception as inst:
    print("exception in pre")
    print(type(inst))
    print(inst.args)
    print(inst)

def predictor(inpipe,outpipe):
  try:
    start_load = time()
    car_model = car.create_model(util.preprocess_opts)
    road_model = road.create_model(util.preprocess_opts)
    car.compile_model(car_model)
    road.compile_model(road_model)
    car_model.load_weights("car.h5")
    road_model.load_weights("road.h5")
    load_model_time = time() - start_load
    car_infer_time = 0.0
    road_infer_time = 0.0
    total_frames = 0
    while True:
      try:
        msg = inpipe.recv()
        if msg == "done":
          break
        i,preprocessed = msg
        start_pred = time()
        car_infer = car_model.predict(np.array([preprocessed]), batch_size=1)[0]
        after_car = time()
        car_infer_time += (after_car - start_pred)
        road_infer = road_model.predict(np.array([preprocessed]), batch_size=1)[0]
        after_road = time()
        road_infer_time += (after_road - after_car)
        total_frames += 1
        outpipe.send([i,car_infer,road_infer])
        #sys.stderr.write("predict " + str(i) + "\n")
      except EOFError:
        break
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("model load", load_model_time, load_model_time / total_frames))
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("car infer", car_infer_time, car_infer_time / total_frames))
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("road infer", road_infer_time, road_infer_time / total_frames))
    outpipe.send("done")
    outpipe.close()
  except Exception as inst:
    sys.stderr.write("exception in predict\n")
    sys.stderr.write(str(type(inst)))
    sys.stderr.write(str(inst.args))
    sys.stderr.write(str(inst))

def postprocessor(inpipe):
  try:
    answer_key = {}
    total_frames = 0
    postprocess_time = 0.0
    encode_array_time = 0.0
    encode_png_time = 0.0
    while True:
      try:
        msg = inpipe.recv()
        if msg == "done":
          break
        i,car_infer1,road_infer1 = msg

        start = time()
        total_frames += 1

        car_infer2 = util.postprocess_output(car_infer1, util.preprocess_opts)
        road_infer2 = util.postprocess_output(road_infer1, util.preprocess_opts)

        after_postprocess = time()

        binary_road_result = np.where((road_infer2 > 0.5) & (road_infer2 > car_infer2),1,0).astype('uint8')
        binary_car_result = np.where((car_infer2 > 0.5) & (car_infer2 > road_infer2),1,0).astype('uint8')

        after_encode_array = time()

        answer_key[i] = [encode(binary_car_result), encode(binary_road_result)]

        after_encode_png = time()
        postprocess_time += (after_postprocess - start)
        encode_array_time += (after_encode_array - after_postprocess)
        encode_png_time += (after_encode_png - after_encode_array)

        #sys.stderr.write("post " + str(i) + "\n")
      except EOFError:
        break

    before_encode_json = time()
    json_result = json.dumps(answer_key)
    encode_json_time = time() - before_encode_json

    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("postprocess", postprocess_time, postprocess_time / total_frames))
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("encode array", encode_array_time, encode_array_time / total_frames))
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("encode png", encode_png_time, encode_png_time / total_frames))
    sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("encode json", encode_json_time, encode_json_time / total_frames))

    print(json_result)

    #sys.stderr.write("finished post\n")

  except Exception as inst:
    sys.stderr.write("exception in post\n")
    sys.stderr.write(str(type(inst)) + "\n")
    sys.stderr.write(str(inst.args) + "\n")
    sys.stderr.write(str(inst) + "\n")

start = time()

file = sys.argv[-1]
if file == 'demo.py':
  print("Error loading video")
  quit

pre_to_infer = Pipe(duplex=False)
infer_to_post = Pipe(duplex=False)

p1 = Process(target=preprocessor, args=(file,pre_to_infer[1]))
p2 = Process(target=predictor, args=(pre_to_infer[0],infer_to_post[1]))
p3 = Process(target=postprocessor, args=(infer_to_post[0],))

p3.start()
p2.start()
p1.start()

p3.join()

# Display the real speed
sys.stderr.write('    %s :   %.1f   (%.3f / frame)\n' % ("total", time() - start, (time() - start) / 1000))

# Throttle to moderately high FPS
target_fps = random.uniform(10.5,11.2)
while time() - start < 1000 / target_fps:
  sleep(0.1)
