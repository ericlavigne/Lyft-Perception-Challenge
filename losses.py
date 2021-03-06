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

def precision(y_true,y_pred):
  true_pos = K.sum(y_true * K.round(y_pred))
  false_pos = K.sum((1 - y_true) * K.round(y_pred))
  return (true_pos + 0.1) / (true_pos + false_pos + 0.1)

def recall(y_true,y_pred):
  true_pos = K.sum(y_true * K.round(y_pred))
  false_neg = K.sum(y_true * K.round(1 - y_pred))
  return (true_pos + 0.1) / (true_pos + false_neg + 0.1)

# Beta is 2 for cars and .5 or road
def f_score(beta):
  def fscore(y_true,y_pred):
    p = precision(y_true,y_pred)
    r = recall(y_true,y_pred)
    return (1.0 + beta*beta) * (p * r) / (beta*beta * p + r)
  return fscore

# Will round num to 0 or 1, but with a continuous and differentiable curve
def round_cont(num):
  return K.tanh(num * 4 - 2.0) * 0.5 + 0.5

def precision_cont(y_true,y_pred):
  true_pos = K.sum(y_true * round_cont(y_pred))
  false_pos = K.sum((1 - y_true) * round_cont(y_pred))
  return (true_pos + 0.1) / (true_pos + false_pos + 0.1)

def recall_cont(y_true,y_pred):
  true_pos = K.sum(y_true * K.round(y_pred))
  false_neg = K.sum(y_true * K.round(1 - y_pred))
  return (true_pos + 0.1) / (true_pos + false_neg + 0.1)

# Returns approximately (1.0 - fscore) but using continuous/differentiable approximation
def f_score_loss(beta):
  def fscore_loss(y_true,y_pred):
    p = precision_cont(y_true,y_pred)
    r = recall_cont(y_true,y_pred)
    return 1.0 - ((1.0 + beta*beta) * (p * r) / (beta*beta * p + r))
  return fscore_loss


def dice(y_true,y_pred):
  """Dice is designed as a continuous and differentiable
     version of the F-score."""
  true_pos = K.sum(y_true * K.abs(y_pred))
  true_neg = K.sum((1 - y_true) * K.abs(1 - y_pred))
  falses = K.sum(K.abs(y_pred - y_true))
  smooth = 10.0
  return (true_pos * 2 + falses + smooth) / (true_pos * 2 + smooth) - 1

def squared_union_over_intersection(y_true, y_pred):
  """Combining ideas from dice loss and mean squared error.
     Calculate union over intersection based on dice loss.
     Perform same calculation for both positive and negative
     classes and add the result so that the metric works for
     both common and rare classes. Calculating inclusion in
     a class based on squared distance rather than distance
     encourages choosing 0.5 rather than guessing 0 or 1
     when correct answer is uncertain. This penalty for
     guessing is similar to the difference between mean
     squared error and mean absolute error."""
  true_pos = K.sum(y_true * K.square(y_pred))
  true_neg = K.sum((1 - y_true) * K.square(1 - y_pred))
  falses = K.sum(K.square(y_pred - y_true))
  smooth = 10.0
  pos_loss = (true_pos + falses + smooth) / (true_pos + smooth)
  neg_loss = (true_neg + falses + smooth) / (true_neg + smooth)
  return pos_loss + neg_loss - 2
