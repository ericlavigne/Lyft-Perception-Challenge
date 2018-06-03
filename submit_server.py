import cv2, sys, json, base64, zmq
from multiprocessing import Process, Pipe
import numpy as np
import random
import car, road, util
from io import BytesIO, StringIO
from time import time, sleep

def encode(array):
  retval, buff = cv2.imencode('.png',array)
  return base64.b64encode(buff).decode("utf-8")

def preprocessor(inpipe,outpipe):
  while True:
    try:
      filename = inpipe.recv()
      start_time = time()
      video = cv2.VideoCapture(filename)
      after_read_video = time()
      read_video_time = after_read_video - start_time
      total_frames = 0
      cropscale_time = 0.0
      send_time = 0.0
      extract_image_time = 0.0
      rgb_convert_time = 0.0
      while True:
        before_extract_image = time()
        frame_available, bgr_frame = video.read()
        if not frame_available:
          break
        total_frames = total_frames + 1
        after_extract_image = time()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        before_preprocess_image = time()
        preprocessed = util.preprocess_input_image(rgb_frame, util.preprocess_opts)
        after_preprocess_image = time()
        outpipe.send([total_frames,preprocessed])
        after_send = time()
        cropscale_time += after_preprocess_image - before_preprocess_image
        send_time += after_send - after_preprocess_image
        extract_image_time += before_preprocess_image - before_extract_image
        rgb_convert_time += before_preprocess_image - after_extract_image
      preprocess_time = time() - start_time
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("preprocessor", preprocess_time, preprocess_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % (" pre:read", read_video_time, read_video_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % (" pre:extract", extract_image_time, extract_image_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % (" pre:rgbconv", rgb_convert_time, rgb_convert_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % (" pre:cropscale", cropscale_time, cropscale_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % (" pre:send", send_time, send_time / total_frames, total_frames))
      outpipe.send(["total_frames", total_frames])

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
    sys.stderr.write('    %s :   %.1f\n' % ("model load", load_model_time))
    while True:
      total_frames = None # Not known yet
      frames_so_far = 0
      car_infer_time = 0.0
      road_infer_time = 0.0
      while True:
        if total_frames is not None:
          if frames_so_far >= total_frames:
            break
          else:
            sys.stderr.write("predict missing frames: " + str(total_frames - frames_so_far) + "\n")
        msg = inpipe.recv()
        if msg[0] == "total_frames":
          total_frames = msg[1]
          #sys.stderr.write("predict received total_frame message: " + str(total_frames) + "\n")
          continue
        frames_so_far += 1
        i,preprocessed = msg
        #sys.stderr.write("predict receives " + str(i) + "\n")
        start_pred = time()
        car_infer = car_model.predict(np.array([preprocessed]), batch_size=1)[0]
        after_car = time()
        car_infer_time += (after_car - start_pred)
        road_infer = road_model.predict(np.array([preprocessed]), batch_size=1)[0]
        after_road = time()
        road_infer_time += (after_road - after_car)
        outpipe.send([i,car_infer,road_infer])
      #sys.stderr.write("End of predict\n")
      #sys.stderr.write("total_frames is " + str(total_frames) + "\n")
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("car infer", car_infer_time, car_infer_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("road infer", road_infer_time, road_infer_time / total_frames, total_frames))
      outpipe.send(["total_frames",total_frames])
  except Exception as inst:
    sys.stderr.write("exception in predict\n")
    sys.stderr.write(str(type(inst)))
    sys.stderr.write(str(inst.args))
    sys.stderr.write(str(inst))

def postprocessor(inpipe,outpipe):
  try:
    while True:
      answer_key = {}
      total_frames = None
      frames_so_far = 0
      postprocess_time = 0.0
      encode_array_time = 0.0
      encode_png_time = 0.0
      while True:
        if total_frames is not None and frames_so_far >= total_frames:
          break
        msg = inpipe.recv()
        if msg[0] == "total_frames":
          total_frames = msg[1]
          continue
        frames_so_far += 1

        i,car_infer1,road_infer1 = msg

        start = time()

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

      before_encode_json = time()
      json_result = json.dumps(answer_key)
      encode_json_time = time() - before_encode_json

      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("postprocess", postprocess_time, postprocess_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("encode array", encode_array_time, encode_array_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("encode png", encode_png_time, encode_png_time / total_frames, total_frames))
      sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("encode json", encode_json_time, encode_json_time / total_frames, total_frames))

      outpipe.send([total_frames, json_result])

  except Exception as inst:
    sys.stderr.write("exception in post\n")
    sys.stderr.write(str(type(inst)) + "\n")
    sys.stderr.write(str(inst.args) + "\n")
    sys.stderr.write(str(inst) + "\n")

sys.stderr.write("Starting warmup...\n")
warmup_start = time()

main_to_pre = Pipe(duplex=False)
pre_to_infer = Pipe(duplex=False)
infer_to_post = Pipe(duplex=False)
post_to_main = Pipe(duplex=False)

p1 = Process(target=preprocessor, args=(main_to_pre[0],pre_to_infer[1]))
p2 = Process(target=predictor, args=(pre_to_infer[0],infer_to_post[1]))
p3 = Process(target=postprocessor, args=(infer_to_post[0],post_to_main[1]))

p3.start()
p2.start()
p1.start()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5555')

main_to_pre[1].send("images/test_video.mp4")
test_frames, test_encoding = post_to_main[0].recv()

sys.stderr.write('Completed warmup in %.3f seconds\n' % ((time() - warmup_start),))

while True:
  filename = socket.recv_string()
  start = time()
  main_to_pre[1].send(filename)
  sys.stderr.write("Processing file: " + filename + "\n")
  frames,encoding = post_to_main[0].recv()

  # Display the real speed
  sys.stderr.write('    %s :   %.1f   (%.3f / frame) for %i frames\n' % ("total", time() - start, (time() - start) / frames, frames))

  # Throttle to moderately high FPS
  target_fps = random.uniform(10.5,11.2)
  while time() - start < frames / target_fps:
    sleep(0.1)

  socket.send_string(encoding)
