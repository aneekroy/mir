import os
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

TRAIN_FILE = 'train.tfrecords'
#VALID_FILE = 'valid.tfrecords'
TEST_FILE = 'test.tfrecords'
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 57
NUM_CHANNELS = 3

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={'image/encoded': tf.FixedLenFeature([], tf.string), 'image/class/label': tf.FixedLenFeature([], tf.int64)})

  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  label = tf.cast(features['image/class/label'], tf.int32)
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
  image = tf.reduce_mean(image, axis=2)
  image= tf.expand_dims(image, axis = 2)
  print(image.get_shape())
  #image = tf.transpose(image)
  #image = tf.squueze(image,[2])
  return image, label

def inputs(eval_data, data_dir, batch_size):
  if eval_data == False:
    filename = [os.path.join(data_dir,TRAIN_FILE)]
  else:
    filename = [os.path.join(data_dir, TEST_FILE)]  
  filename_queue = tf.train.string_input_producer(filename)
  image, label = read_and_decode(filename_queue)
  images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=32, capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
  return images, sparse_labels
