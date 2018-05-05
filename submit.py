import sys, skvideo.io, json, base64
import numpy as np
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

for i, rgb_frame in enumerate(video, start=1):
    red = rgb_frame[:,:,0]
    binary_car_result = np.where(red>250,1,0).astype('uint8')
    binary_road_result = binary_car_result = np.where(red<20,1,0).astype('uint8')
    answer_key[i] = [encode(binary_car_result), encode(binary_road_result)]

print (json.dumps(answer_key))
