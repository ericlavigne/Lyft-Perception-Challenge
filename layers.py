from keras.layers import Concatenate
from keras.layers.convolutional import AveragePooling2D, Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

def conv(previous_layer, depth, kernel):
  layer1 = Conv2D(depth, kernel, padding='same')(previous_layer)
  layer2 = BatchNormalization()(layer1)
  layer3 = Activation('elu')(layer2)
  return layer3

def inception(prev, depth):
  """Based on second InceptionV3 module whose output depth is 288.
     Scale all component depths based on ratio between desired depth
     and 288. Prevent any component depth from dropping below 16."""
  scale48 = max(16, int(48 * depth / 288))
  scale64 = max(16, int(64 * depth / 288))
  scale96 = max(16, int(96 * depth / 288))

  layer_1 = conv(prev, scale64, 1)

  layer_2_1 = conv(prev, scale48, 1)
  layer_2_2 = conv(layer_2_1, scale64, 5)

  layer_3_1 = conv(prev, scale64, 1)
  layer_3_2 = conv(layer_3_1, scale96, 3)
  layer_3_3 = conv(layer_3_2, scale96, 3)

  layer_4_1 = AveragePooling2D(pool_size=3, strides=1, padding='same')(prev)
  layer_4_2 = conv(layer_4_1, scale64, 1)

  return Concatenate()([layer_1, layer_2_2, layer_3_3, layer_4_2])
