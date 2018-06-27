import os
import numpy as np
import scipy.io.wavfile as wav
import random
from random import shuffle
import tensorflow as tf

def load_data():
	'''
	Associate the samples with their corresponding labels.
	There are a total of 30 labels.
	'''
	workpath = '/home/soms/EmotionMusic/emotion/Feature_Folder/train.wavelet.text'
	os.chdir(workpath)
	print("...Loading data...\n")
	classcount = 30	
	samples = []
	labels = []
	for dirpath, dirnames, filenames in os.walk(workpath):
		for filename in filenames:
			fileindex = filename[:filename.find('.')]
			label = (int(fileindex) - 1)/classcount
			with open(filename, 'r') as f:
				cols = {}
				lines = f.readlines()
				sample = []
				for line in lines:
					line = line.split(',')
					c = 0
					for l in line:
						if c not in cols:
							cols[c]=[l]
						else:
							cols[c].append(l)
						c+=1
				for c in cols:
					sample.append(cols[c])
			samples.append(sample)
			labels.append(label)

	'''
	Divide the samples into training and test.
	Since there are 30 for each label, we take 25 for train,
	and 5 for test from each label.
	'''


	sampledict = {}
	for c in range(0, len(samples)):
		label = labels[c]
		sample = samples[c]
		if label not in sampledict:
			sampledict[label] = [sample]
		else:
			sampledict[label].append(sample)

	trainsetx = []
	trainsety = []
	testsetx = []
	testsety = []

	for label in sampledict:
		for x in sampledict[label][:25]:
			trainsetx.append(x)
			l=[0]*12
			l[label]=1
			trainsety.append(l)
		for x in sampledict[label][25:]:
			testsetx.append(x)
			l=[0]*12
			l[label]=1
			testsety.append(l)
	a = np.asarray(trainsetx)
	b = np.asarray(trainsety)
	print a.shape, b.shape
	return trainsetx, trainsety, testsetx, testsety		


def train_model():
	NUM_EXAMPLES = 300
	train_input, train_output, test_input, test_output = load_data()
	print('Data loaded...')

	data = tf.placeholder(tf.float32, [None, 480, 600])
	target = tf.placeholder(tf.float32, [None, 12])
	num_hidden = 128
	#num_hidden2 = 16
	cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple = True)
	#cell2 = tf.nn.rnn_cell.LSTMCell(num_hidden2, state_is_tuple = True)
	#stack = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple = True)
	output,_ = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)
	output = tf.transpose(output, [1,0,2])
	last = tf.gather(output, int(output.get_shape()[0]) - 1)
	W = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])], stddev = 0.1))
	b = tf.Variable(tf.constant(0.0, shape = [target.get_shape()[1]]))
	logits = tf.matmul(last,W) + b
	prediction = tf.nn.softmax(logits)
	cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
	optimizer = tf.train.MomentumOptimizer(0.001, 0.9)
	minimize = optimizer.minimize(cross_entropy)
	mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
	error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

	init_op = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init_op)

	batch_size = 10
	no_of_batches = int(len(train_input)) / batch_size
	epochs = 50
	besterr = 1.0
	path = "/home/soms/TensorFlowModels"
	os.chdir(path)
	wt = open("weight.txt", "w")
	bs = open("bias.txt", "w")
	for i in range(epochs):
		ptr = 0
		s = 0
		for j in range(no_of_batches):
			inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
			ptr += batch_size
			sess.run(minimize, {data: inp, target: out})
			incorrect = sess.run(error, {data: inp, target: out})
			s += incorrect
		print "Epoch ", str(i)
		s /= no_of_batches
		print("Train Error {:3.2f}%".format(100 * s))
		print "\n"
		testerr = sess.run(error, {data:test_input, target:test_output})
		print("Test Error {:3.2f}%".format(100 * testerr))
		#Save model params
		if testerr<besterr:
			'''
			for x in range(num_hidden):
				for y in range(12):
					wt.write(str(W[x,y])+' ')
			for x in range(12):
				bs.write(str(b[x])+' ')
			'''
			besterr = testerr

		print "\n"
	print "Best model test error: " + str(besterr)
	sess.close()


if __name__ == '__main__':
	train_model()