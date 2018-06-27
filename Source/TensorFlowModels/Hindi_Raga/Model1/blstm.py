from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from math import *
from six.moves import urllib
import tensorflow as tf
import reader

from tensorflow.python.ops import rnn, rnn_cell

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1635,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/soms/EmotionMusic/ModelRaga',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the data set.
#IMAGE_SIZE = 32
NUM_CLASSES = 24
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6476
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1635

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 800.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.005  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.005       # Initial learning rate.

TOWER_NAME = 'tower'

def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name, shape, tf.random_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'Data')
  batch_size = FLAGS.batch_size if eval_data else 32
  images, labels = reader.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def put_kernels_on_grid(kernel, pad = 1):
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i ==0:
        return (i, int(n / i))
  (grid_y, grid_X) = factorization(kernel.get_shape()[3].value)

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)

  kernel1 = (kernel - x_min) / (x_max - x_min)

  x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode = 'CONSTANT')

  Y = kernel1.get_shape()[0] + 2 * pad
  X = kernel1.get_shape()[1] + 2 * pad

  channels = kernel1.get_shape()[2]

  x2 = tf.transpose(x1, (3, 0 ,1, 2))
  x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_y, X, channels]))
  x4 = tf.transpose(x3, (0, 2, 1, 3))
  x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_y, channels]))
  x6 = tf.transpose(x5, (2, 1, 3, 0))
  x7 = tf.transpose(x6, (3, 0, 1, 2))

  return x7

def inference(images, eval = False):
  NUM_HIDDEN = 128
  W = _variable_with_weight_decay('weight',[2*NUM_HIDDEN, NUM_CLASSES], 0.001, 0.01)
  b = _variable_with_weight_decay('bias', [NUM_CLASSES], 0.001, 0)
  s = images.get_shape()
  n_batches, n_steps, n_features = int(s[0]), int(s[1]), int(s[2])
  #print(n_batches, n_steps, n_features)
  inputs = tf.reshape(images, [-1, n_features])
  inputs = tf.split(0, n_steps, inputs)

  fw_cell = rnn_cell.LSTMCell(NUM_HIDDEN, forget_bias = 1.0, state_is_tuple = True)
  bw_cell = rnn_cell.LSTMCell(NUM_HIDDEN, forget_bias = 1.0, state_is_tuple = True)

  try:
    outputs,_,_ = rnn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype = tf.float32)
  except Exception:
    outputs = rnn.bidirectional_rnn(fw_cell, bw_cell, x, dtype = tf.float32)
  logits = tf.matmul(outputs[-1], W) + b
  return logits

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
   	opt = tf.train.AdamOptimizer()
  	grads = opt.compute_gradients(total_loss)

 	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
     	tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
