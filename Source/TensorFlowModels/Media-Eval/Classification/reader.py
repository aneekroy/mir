import os
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

TRAIN_FILE = 'train.tfrecords'
#VALID_FILE = 'valid.tfrecords'
TEST_FILE = 'test.tfrecords'
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 56
NUM_CHANNELS = 1

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={'image_raw': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  print(image.get_shape())
  label = tf.cast(features['label'], tf.int32)
  print(image.get_shape())
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
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
