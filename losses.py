import tensorflow as tf
from keras import backend as K

def weighted_binary_crossentropy(weight):
  """Higher weights increase the importance of examples in which
     the correct answer is 1. Higher values should be used when
     1 is a rare answer. Lower values should be used when 0 is
     a rare answer. In the Lyft training data, a weight of 34.6
     compensates for cars only representing 2.8% of all pixels."""
  return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

def weighted_mean_squared_error(weight):
  """Higher weights increase the importance of examples in which
     the correct answer is 1. Higher values should be used when
     1 is a rare answer. Lower values should be used when 0 is
     a rare answer."""
  def weighted_mse_aux(y_true, y_pred):
    weights = weight * y_true + (1 - y_true)
    return tf.losses.mean_squared_error(y_true, y_pred, weights)
  return weighted_mse_aux

def balanced_binary_mean_squared_error(y_true, y_pred):
  """Assuming that y_true is a mix of 1 and 0, automatically
     applies appropriate weighting to pixels so that each
     category contributes equally to the loss. Aside from
     the weighting, this loss is mean-squared error. The
     smooth parameter prevents very extreme weights when
     one class represents less than .1% of the total."""
  count_ones = K.sum(K.abs(y_true))
  count_zeros = K.sum(K.abs(1.0 - y_true))
  smooth = 0.001
  count_total = count_zeros + count_ones
  weight_ones = count_total * (1 + smooth) / (2 * count_ones + smooth * count_total)
  weight_zeros = count_total * (1 + smooth) / (2 * count_zeros + smooth * count_total)
  return K.sum((weight_ones * y_true + weight_zeros * (1 - y_true)) * K.square(y_true - y_pred)) / count_total
