from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import convnet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/soms/EmotionMusic/MediaEval/Evaluation_Logs',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/soms/EmotionMusic/MediaEval/Training_Logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1 * 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 6290,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")



def eval_once(saver, summary_writer, logits, labels, loss, r_y, r_e, summary_op):
  with tf.Session() as sess:
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
      mean_accuracy = 0
      m = 999.0
      M = -999.0
      S_accuracy = 0
      S_loss = 0
      for index in range(10):
        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0.  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        loss_sum = 0.
        c = 0
        while step < num_iter and not coord.should_stop():
          logs, labs, loss_val, squared_error, mean_error = sess.run([logits, labels, loss, r_y, r_e])
          for i in range(3):
	  	print("Actual val:%.2f Predicted val:%.2f"%(labs[i], logs[i]))
          loss_sum += loss_val
          r_squared = 1 - float(squared_error) / mean_error
          print("R_y:%.2f R_e:%.2f R_squared:%.2f Loss:%.2f"%(squared_error, mean_error, r_squared, loss_val))
          true_count += r_squared
          step += 1
        
        S_accuracy += true_count
        S_loss += loss_sum
        if true_count < m:
          m = true_count
        if true_count > M:
          M = true_count
      mean_accuracy = S_accuracy/10
      mean_loss = S_loss/10
      print("Average loss = %.2f   Average r-squared = %.2f   Min r-squared = %.2f   Max r-squared = %.2f"%(mean_loss, mean_accuracy, m, M))
      #print('%s: precision @ 1 = %.3f loss = %.2f' % (datetime.now(), precision, mean_loss))
      

      
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='R-square', simple_value=mean_accuracy)
      summary.value.add(tag='total_loss', simple_value=mean_loss)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():

  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == 'test'
    images, labels = convnet.inputs(eval_data = True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = convnet.inference(images, eval = True)
    loss = convnet.loss(logits, labels)
    # Calculate r-squared measure
    R_y = tf.reduce_sum(tf.square(labels - tf.squeeze(logits)))
    R_e = tf.reduce_sum(tf.square(labels - tf.reduce_mean(tf.squeeze(logits))))


    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(convnet.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, logits, labels, loss, R_y, R_e, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
  evaluate()
