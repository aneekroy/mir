from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import sys
import math
import time
import os
import numpy as np
import tensorflow as tf

import convnet

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 56
NUM_CLASSES = 4

ALPHA_MAP = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H"}

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/soms/EmotionMusic/MediaEval_Classification/Evaluation_Logs',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/soms/EmotionMusic/MediaEval_Classification/Training_Logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1.5 * 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 125,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

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
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
  return image, label

def input(sess, filename_queue):
  images = []
  labels = []
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  image ,label = read_and_decode(filename_queue)
  im, lb = sess.run([image, label])
  images.append(im)
  labels.append(lb)
  while True:
    im,lb = sess.run([image, label])
    if not np.array_equal(im, np.asarray(images[0])):
      images.append(im)
      labels.append(lb)
    else:
      break
  #images = tf.convert_to_tensor(images)
  #labels = tf.convert_to_tensor(labels)
  images = np.asarray(images)
  labels = np.asarray(labels)
  return images, labels


def eval_once(sess, saver, logits, labels, filename, predictionfile, sequencefile):
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
      
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print("Global step %s"%global_step)
  else:
    print('No checkpoint file found')
    return

    # Start the queue runners.
  coord = tf.train.Coordinator()
  try:
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    probs = sess.run(tf.argmax(logits, 1))
    preds = {}
    for i in range(NUM_CLASSES):
      preds[i] = 0
    for i in range(len(probs)): 
        preds[probs[i]] += 1
    actual_label = labels[0]
    predictionfile.write(filename + "\t")
    sequencefile.write(filename + "\t")
    for i in range(len(probs)):
      sequencefile.write("%s "%(ALPHA_MAP[probs[i]]))
    sequencefile.write("  " + str(actual_label) + "\n")
    s = 0
    for i in range(NUM_CLASSES):
      s += preds[i]
    for i in range(NUM_CLASSES):
      predictionfile.write("%d(%.2f)\t"%(preds[i], float(preds[i])/s))
    predictionfile.write(str(actual_label) + "\n")
  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
  predicted_max = [0]
  for i in range(NUM_CLASSES):
    if preds[i] > preds[predicted_max[0]]:
      predicted_max = [i]
    elif preds[i] == preds[predicted_max[0]]:
    	predicted_max.append(i)
  prec_song = 0
  if labels[0] in predicted_max:
    prec_song = 1
  tot_clips = len(probs)
  prec_clips = 0
  for i in range(len(probs)):
    if probs[i] == labels[0]:
      prec_clips += 1
  return prec_song, prec_clips, tot_clips


def evaluate(ind = None):

  testdir = "/home/soms/EmotionMusic/MediaEval_Classification/Data/Test"
  index = ind if ind is not None else 0
  while True:
    with tf.Graph().as_default() as g:
      predictionfile = open("/home/soms/EmotionMusic/MediaEval_Classification/Outputs/output_" + str(index) + ".txt", "w")
      sequencefile = open("/home/soms/EmotionMusic/MediaEval_Classification/Outputs/output_sequence_" + str(index) + ".txt", "w")
      with tf.Session() as sess:  
        print("Index %d"%index)
        for _,_,files in os.walk(testdir):
          cnt = 1
          prec_song = 0
          prec_clips = 0
          tot_clips = 0   
          for file in files:  
            filename = os.path.join(testdir, file)
            filename_queue = tf.train.string_input_producer([filename])
            images, labels = input(sess, filename_queue)
            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = convnet.inference(images, eval = True)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(convnet.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            p_song, p_clips, t_clips = eval_once(sess, saver, logits, labels, file, predictionfile, sequencefile)
            prec_song += p_song
            prec_clips += p_clips
            tot_clips += t_clips
            print("Count %d"%cnt)
            cnt += 1
        prec_song /= len(files)
        prec_clips /= tot_clips
        print("..........................................................")
        print("Total Song# %d"%len(files))
        print("Total Clips# %d"%tot_clips)
        print("Song level accuracy %.2f Clip level accuracy %.2f"%(prec_song, prec_clips))
        print("..........................................................")
    time.sleep(FLAGS.eval_interval_secs)
    index += 1

if __name__ == '__main__':
  try:
    ind = int(sys.argv[1])
  except Exception:
    print("Usage: python model_eval_classwise.py index")
    sys.exit(0)
  evaluate(ind)
