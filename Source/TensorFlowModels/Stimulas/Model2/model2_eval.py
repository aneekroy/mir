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

tf.app.flags.DEFINE_string('eval_dir', '/home/soms/EmotionMusic/Model2/Evaluation_Logs',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/soms/EmotionMusic/Model2/Training_Logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1.5 * 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 2126,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")



def eval_once(saver, summary_writer, logits, labels, top_k_op, conf_mat, loss, summary_op):
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
      mean_precision = 0
      m = 1.0
      M = 0.0
      for index in range(10):
        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        loss_sum = 0
        c = 0
        cf = []
        while step < num_iter and not coord.should_stop():
          confusion_matrix, predictions = sess.run([conf_mat, top_k_op])
          cf.append(confusion_matrix)
          #loss_val = sess.run(loss)
          #loss_sum += loss_val
          true_count += np.sum(predictions)
          step += 1
        # Compute precision @ 1.
        precision = true_count / total_sample_count
        mean_loss = loss_sum / step
        mean_precision += precision
        if precision < m:
          m = precision
        if precision > M:
          M = precision
      mean_precision /= 10
      prec = {}
      rec = {}
      fm = {}
      confusion_matrix = np.mean(np.asarray(cf), 0)
      print("Confusion Matrix")
      print(confusion_matrix)
      for i in range(len(confusion_matrix)):
        if np.sum(confusion_matrix, 1)[i] != 0.0:
          prec[i] = float(confusion_matrix[i][i])/(np.sum(confusion_matrix, 1)[i])
        else:
          prec[i] = 0.0
        if np.sum(confusion_matrix, 0)[i] != 0.0:
          rec[i] = float(confusion_matrix[i][i])/(np.sum(confusion_matrix, 0)[i])
        else:
          rec[i] = 0.0
        if (prec[i] + rec[i]) != 0.0:
          fm[i] = float(2 * prec[i] * rec[i])/(prec[i] + rec[i])
        else:
          fm[i] = 0.0
        print("Class#%d precision: %.2f recall: %.2f f-measure: %.2f"%(i, prec[i], rec[i], fm[i]))

      print("\nAverage precision %.3f Min precision %.3f Max precision %.3f"%(mean_precision, m, M))
      #print('%s: precision @ 1 = %.3f loss = %.2f' % (datetime.now(), precision, mean_loss))
      

      
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=mean_precision)
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

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    #Compute confusion matrix
    conf_mat = tf.contrib.metrics.confusion_matrix(tf.arg_max(logits, 1), tf.cast(labels, tf.int64))

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(convnet.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, logits, labels, top_k_op, conf_mat, loss, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
  evaluate()