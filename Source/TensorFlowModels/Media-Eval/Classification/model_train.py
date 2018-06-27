from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import os
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import convnet
import reader
import model_eval

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/soms/EmotionMusic/MediaEval_Classification/Training_Logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

  
def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels.
    images, labels = convnet.inputs(eval_data = False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = convnet.inference(images)

    # Calculate loss.
    loss = convnet.loss(logits, labels)

    # Calculate accuracy
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = convnet.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Load previously stored model from checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print("Loading from checkpoint.Global step %s"%global_step)
    else:
      print("No checkpoint file found...Creating a new model...")

    stepfile = "/home/soms/EmotionMusic/MediaEval_Classification/stepfile.txt" 
    if not os.path.exists(stepfile):
      print("No step file found.")
      step = 0
    else:
      f = open(stepfile, "r")
      step = int(f.readlines()[0])
      print("Step file step %d"%step) 

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    while step < FLAGS.max_steps:
        #print(images.eval(session = sess).shape)
        start_time = time.time()
        _, loss_value, predictions = sess.run([train_op, loss, top_k_op])
        duration = time.time() - start_time

        def signal_handler(signal, frame):
          f = open(stepfile, 'w')
          f.write(str(step))
          print("Step file written to.")
          sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        if step % 10 == 0:
          num_examples_per_step = 25
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          accuracy = float(np.sum(predictions))/num_examples_per_step
          format_str = ('%s: step %d, loss = %.2f accuracy@1 = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
          print (format_str % (datetime.now(), step, loss_value, accuracy, examples_per_sec, sec_per_batch))

        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
          model_eval.evaluate(mode = 1)
        step += 1

if __name__ == '__main__':
  train()
