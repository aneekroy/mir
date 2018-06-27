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

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1968,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/satyaki/Emotion/Soundtrack/Model2_SongWise',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the data set.
#IMAGE_SIZE = 32
NUM_CLASSES = 4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 7425
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3181

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 800.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.005  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.005       # Initial learning rate.

TOWER_NAME = 'tower'

def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'Data')
  batch_size = FLAGS.batch_size if eval_data else 16
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
  x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_y, X, channels]))
  x4 = tf.transpose(x3, (0, 2, 1, 3))
  x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_y, channels]))
  x6 = tf.transpose(x5, (2, 1, 3, 0))
  x7 = tf.transpose(x6, (3, 0, 1, 2))

  return x7

def inference(images, eval = False):
  
  keep1 = 1.0
  keep2 = 1.0
  keep3 = 1.0
  keep4 = 1.0
  keep5 = 1.0
  #keep5 = 1.0

  if eval == False:
    keep1 = 0.75
    keep2 = 0.75
    keep3 = 0.75
    keep4 = 0.75
    keep5 = 0.5

  # Add  images
  if eval == False: 
    disp_images = tf.transpose(images, (1, 2, 3, 0))
    grid = put_kernels_on_grid(disp_images)
    tf.summary.image("images", grid, max_outputs = 1)
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    try:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 1, 32], stddev=5e-2, wd=0.0)
    except ValueError:
      scope.reuse_variables()
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 1, 32], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    #print (images.get_shape, conv.get_shape())
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    try:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 32], stddev=5e-2, wd=0.0)
    except ValueError:
      scope.reuse_variables()
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 32], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    #print (images.get_shape, conv.get_shape())
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)


  # pool1
  pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


  # drop1
  drop1 = tf.nn.dropout(pool1, keep_prob = keep1, name='drop1')
  _activation_summary(drop1)
  
  # norm1
  #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
  
  # conv3
  with tf.variable_scope('conv3') as scope:
    try:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 64], stddev=5e-2, wd=0.0)
    except ValueError:
      scope.reuse_variables()
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(drop1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    try:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
    except ValueError:
      scope.reuse_variables()
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # pool2
  pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # drop2
  drop2 = tf.nn.dropout(pool2, keep_prob = keep2, name='drop2')
  _activation_summary(drop2)

  # norm2
  #norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

  # conv5
  with tf.variable_scope('conv5') as scope:
    try:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
    except ValueError:
      scope.reuse_variables()
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(drop2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)
  
  # pool3
  pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # drop3
  drop3 = tf.nn.dropout(pool3, keep_prob = keep3, name='drop3')
  _activation_summary(drop3)
  
  # conv6
  with tf.variable_scope('conv6') as scope:
    try:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
    except ValueError:
      scope.reuse_variables()
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(drop3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv6)
  
  # pool4
  pool4 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

  # drop4
  drop4 = tf.nn.dropout(pool4, keep_prob = keep4, name='drop4')
  _activation_summary(drop4)
  # local3
  with tf.variable_scope('local1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    batch_size = 16
    if eval == True:
      #batch_size = FLAGS.batch_size
      batch_size = images.shape[0]
    reshape = tf.reshape(drop4, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    try:
      weights = _variable_with_weight_decay('weights', shape=[dim, 256], stddev=0.04, wd=0.1)
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    except ValueError:
      scope.reuse_variables()
      weights = _variable_with_weight_decay('weights', shape=[dim, 256], stddev=0.04, wd=0.1)
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local1)

  # drop5
  drop5 = tf.nn.dropout(local1, keep_prob = keep5, name='drop5')
  _activation_summary(drop5)

  # local4
  #with tf.variable_scope('local4') as scope:
  #  try:
  #    weights = _variable_with_weight_decay('weights', shape=[512, 256], stddev=0.04, wd=0.1)
  #    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
  #  except ValueError:
  #    scope.reuse_variables()
  #    weights = _variable_with_weight_decay('weights', shape=[512, 256], stddev=0.04, wd=0.1)
  #    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
  #  local4 = tf.nn.relu(tf.matmul(drop3, weights) + biases, name=scope.name)
  #  _activation_summary(local4)

  # drop4
  #drop4 = tf.nn.dropout(local4, keep_prob = keep4, name='drop4')
  #_activation_summary(drop4)

  '''
  # local5
  with tf.variable_scope('local5') as scope:
    try:
      weights = _variable_with_weight_decay('weights', shape=[256, 64], stddev=0.04, wd=0.01)
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    except ValueError:
      scope.reuse_variables()
      weights = _variable_with_weight_decay('weights', shape=[256, 64], stddev=0.04, wd=0.01)
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(drop4, weights) + biases, name=scope.name)
    _activation_summary(local5)

  drop5 = tf.nn.dropout(local5, keep_prob = keep5, name='drop5')
  _activation_summary(drop5)
  '''

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    try:
      weights = _variable_with_weight_decay('weights', [256, NUM_CLASSES], stddev=1/256.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    except ValueError:
      scope.reuse_variables()
      weights = _variable_with_weight_decay('weights', [256, NUM_CLASSES], stddev=1/256.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(drop5, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
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
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

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
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
     	tf.summary.histogram(var.op.name + '/gradients', grad)

  # Add conv1 filter images 
  with tf.variable_scope("conv1"):
    tf.get_variable_scope().reuse_variables()
    weights = tf.get_variable("weights")
    grid = put_kernels_on_grid(weights)
    tf.summary.image("conv1/features", grid, max_outputs = 1)

  # Add conv2 filter images 
  with tf.variable_scope("conv2"):
    tf.get_variable_scope().reuse_variables()
    weights = tf.get_variable("weights")
    grid = put_kernels_on_grid(weights[:, :, :1, :])
    tf.summary.image("conv2/features", grid, max_outputs = 1)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op