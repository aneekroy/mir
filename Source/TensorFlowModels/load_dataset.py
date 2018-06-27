import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

test_set_size = 1000
IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 1290
NUM_CHANNELS = 1
BATCH_SIZE = 16
TARGET_SIZE = 24


def prep_label_file():
	path = '/home/soms/EmotionMusic/Spec_Images'
	labelfilepath = '/home/soms/EmotionMusic/image-label-file.txt'
	f = open(labelfilepath, 'w')
	dataset = []
	for _, dirs, _ in os.walk(path):
		label = 0
		for dir in dirs:
			print("Label %d loading started"%label)
			c = 0
			for _, _ , files in os.walk(path + '/' + dir):
				for filename in files:
					c += 1
					filepath = path+'/'+dir+'/'+filename
					f.write(filepath + ':-' + str(label) + '\n')
					#dataset.append(np.genfromtxt(filepath, delimiter = ',').T)
			print("Label %s loaded. %d files loaded"%(dir, c))
			label += 1

def encode_label(label):
	return int(label)

def read_label_file(file):
	f = open(file, "r")
	filepaths = []
	labels = []
	c = 0
	for line in f.readlines():
		c += 1
		try:
			filepath, label = line.split(":-")
			label = int(label)
		except Exception:
			print("Error in line %d: %s"%(c,line))
		filepaths.append(filepath)
		labels.append(label)
	return filepaths, labels

def convert_to_one_hot(data):
	targets = []
	for d in data:
		one_hot = [0]*TARGET_SIZE
		one_hot[d] = 1
		targets.append(one_hot)
	return np.asarray(targets)

train_image_batch = None
train_label_batch = None
test_image_batch = None
test_label_batch = None

def prep_data():
	path = '/home/soms/EmotionMusic/image-label-file.txt'
	#print("...Loading data...")
	filepaths, labels = read_label_file(path)
	data = ops.convert_to_tensor(filepaths, dtype=dtypes.string)
	labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)
	#print("...Partitioning data into train and test set...")
	partitions = [0] * len(filepaths)
	partitions[:test_set_size] = [1] * test_set_size
	random.shuffle(partitions)
	train_data, test_data = tf.dynamic_partition(data, partitions, 2)
	train_labels, test_labels = tf.dynamic_partition(labels, partitions, 2)
	#print("...Creating input queues...")
	train_input_queue = tf.train.slice_input_producer([train_data, train_labels], shuffle=True)
	test_input_queue = tf.train.slice_input_producer([test_data, test_labels], shuffle=True)
	#print("...Processing data...")
	file_content = tf.read_file(train_input_queue[0])
	train_image = tf.image.decode_jpeg(file_content, channels = NUM_CHANNELS)
	train_label = train_input_queue[1]

	file_content = tf.read_file(test_input_queue[0])
	test_image = tf.image.decode_jpeg(file_content, channels = NUM_CHANNELS)
	test_label = test_input_queue[1]

	train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
	test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

	global train_image_batch, train_label_batch, test_image_batch, test_label_batch
	# collect batches of images before processing
	train_image_batch, train_label_batch = tf.train.batch([train_image, train_label], batch_size=BATCH_SIZE, num_threads=5)
	test_image_batch, test_label_batch = tf.train.batch([test_image, test_label], batch_size=BATCH_SIZE, num_threads=5)


def load_data():
#print("...Input pipeline ready...")
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		inputs = sess.run(train_image_batch).squeeze(axis=(3,))
		inputs = inputs.transpose([0,2,1])
		targets = sess.run(train_label_batch)
		targets = convert_to_one_hot(targets)
		coord.request_stop()
		coord.join(threads)
		sess.close()
		return inputs, targets 
	                                    

if __name__ == "__main__":
	load_data()