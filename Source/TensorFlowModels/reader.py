import numpy as np
import os
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

FILE = '/home/soms/EmotionMusic/Model2_SongWise/Data/Test/test_020.tfrecords'
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 56

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={'image_raw': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  label = tf.cast(features['label'], tf.int32)
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
  #image = tf.transpose(image)
  #image = tf.squueze(image,[2])
  return image, label

if __name__ == '__main__':
  filename = FILE  
  filename_queue = tf.train.string_input_producer([filename])
  image, label = read_and_decode(filename_queue)
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images=[]
    labels=[]
    im,lb=sess.run([image, label])
    images.append(im)
    labels.append(lb)
    while True:
      im,lb = sess.run([image, label])
      if not np.array_equal(im, np.asarray(images[0])):
        images.append(im)
        labels.append(lb)
      else:
        break
    print(np.asarray(images).shape, np.asarray(labels).shape)