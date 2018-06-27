from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


def inference(inputs, n_input, n_steps, n_hidden, n_classes):
	W = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
	b = tf.Variable(tf.random_normal([n_classes]))
	inputs = tf.reshape(inputs, [-1, n_input])
	inputs = tf.split(0, n_steps, inputs)

	fw_cell = rnn_cell.LSTMCell(n_hidden, forget_bias = 1.0, state_is_tuple = True)
	bw_cell = rnn_cell.LSTMCell(n_hidden, forget_bias = 1.0, state_is_tuple = True)

	try:
		outputs,_,_ = rnn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype = tf.float32)
	except Exception:
		outputs = rnn.bidirectional_rnn(fw_cell, bw_cell, x, dtype = tf.float32)
	return tf.matmul(outputs[-1], W) + b

def loss(logits, labels):
	labels = tf.to_int64(labels)
  	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  	loss = tf.reduce_mean(cross_entropy)
  	return loss

def optimizer(loss, learning_rate):																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																								learning_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
	return optimizer

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
  	return tf.reduce_mean(tf.cast(correct, tf.int32))

																																																																																																																																																																																																																																																																																																																																																																																																				
with tf.Session() as sess:
	sess.run(init)
	step = 1
	#train_input, train_output, test_input, test_output = load_data()
	while step < no_of_iterations:
		ptr = 0
		j = 0
		while ptr <= no_of_examples:
			ptr += batch_size
			j += 1
			inputs, targets = load_data()
			print("...Minibatch loaded...")
			#print targets
			#ptr += batch_size
			#inputs = inputs.reshape((batch_size, n_steps, n_input))
			sess.run(optimizer, feed_dict = {x:inputs, y:targets})

			if step % output_step == 0:
				acc = sess.run(accuracy, feed_dict = {x:inputs, y:targets})
				loss = sess.run(cost, feed_dict= {x:inputs, y:targets})
				print("Iteration " + str(step) + "Minibatch " +  str(j) + " Loss = " + "{:.6f}".format(loss) + " Accuracy= " + "{:.6f}".format(acc))
		step += 1
	print("...Training complete...")
	acc = sess.run(accuracy, feed_dict = {x:test_input, y:test_output})
	print("Test accuracy :" + "{:.6f}".format(acc))
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			