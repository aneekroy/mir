import numpy as np
import keras
import math
import os
import sys
import signal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Bidirectional, TimeDistributedDense, Merge, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import regularizers
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
import keras.backend as K
from read_data import *
import matplotlib.pyplot as plt
from itertools import product
from functools import partial

def w_categorical_cross_entropy(y_true, y_pred, weights):
	nb_cl = len(weights)
	final_mask = K.zeros_like(y_pred[:, 0])
	y_pred_max = K.max(y_pred, axis=1)
	y_pred_max = K.expand_dims(y_pred_max, 1)
	y_pred_max_mat = K.equal(y_pred, y_pred_max)
	for c_p, c_t in product(range(nb_cl), range(nb_cl)):
		final_mask += (K.cast(weights[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[:, c_p], K.floatx())*K.cast(y_true[:,c_t], K.floatx()))
	return K.categorical_crossentropy(y_pred, y_true) * final_mask


def show_results(model, testrepo, history = None):
	scores = model.evaluate(x_test, y_test, verbose = 1)
	print("Accuracy: %.2f%%"%(scores[1]*100))

	predictions = model.predict_classes(x_test, verbose = 1)
	index = 0
	for p, a in zip(predictions, y_test):
		print("File_Segment:" + testrepo[index] + " Actual:" + str(np.argmax(a)) + " Predicted:" + str(p))
		index += 1
	segment_indices = {}
	for index in range(len(testrepo)):
		filename = testrepo[index]
		if filename in segment_indices:
			segment_indices[filename].append(index)
		else:
			segment_indices[filename] = [index]
	print("Prediction length:%d Actual_length:%d FileRepo_length:%d"%(len(predictions), len(y_test), len(testrepo)))
	whole_set_accuracy = 0.0
	for s in segment_indices:
		cnt = {}
		indices = segment_indices[s] 
		actual = np.argmax(y_test[indices[0]])
		for index in indices:
			if predictions[index] in cnt:
				cnt[predictions[index]] += 1
			else:
				cnt[predictions[index]] = 1
		M = 0
		predicted = 0
		for c in cnt:
			if cnt[c] > M:
				M = cnt[c]
				predicted = c
		if predicted == actual:
			whole_set_accuracy += 1
	whole_set_accuracy = float(whole_set_accuracy) / len(segment_indices.keys())
	print("Whole set accuracy:%.3f"%whole_set_accuracy) 
	
	if history is not None:
		# Plots
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model_accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.show()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model_loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.show()


def create_class_weight(labels_dict,mu=0.15):
    M = np.max(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        class_weight[key] = float(M) / labels_dict[key] 

    return class_weight


seed = 7
np.random.seed(7)
DATA_PATH = "/home/satyaki/Emotion/bimodal/Features"
LABEL_PATH = "/home/satyaki/Emotion/bimodal/annotations.csv"

x_train, y_train, x_test, y_test, testrepo, c0, c1, c2, c3 = load_dataset(DATA_PATH, LABEL_PATH)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

labels_dict = {0:c0, 1:c1, 2:c2, 3:c3}

class_weights = create_class_weight(labels_dict)

for c in class_weights:
	print(str(c) + ":" + str(labels_dict[c]) + ":" + str(class_weights[c]))

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)

model = Sequential()

# Gaussian Noise
model.add(GaussianNoise(input_shape = x_train.shape[1:], sigma = 0.5))

# ConvPool1
model.add(Convolution2D(50, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Dropout(0.35))

# ConvPool2
model.add(Convolution2D(75, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(MaxPooling2D(pool_size = (3, 3), strides = (3, 3)))

model.add(Dropout(0.35))


# Reshape
model.add(Reshape((9, 5*75)))

# Bidirectional LSTM 1
model.add(Bidirectional(LSTM(output_dim = 100, return_sequences = True)))

model.add(Dropout(0.5))

# Bidirectional LSTM 2
model.add(Bidirectional(LSTM(output_dim = 75, return_sequences = False)))

model.add(Dropout(0.5))

# Softmax
model.add(Dense(num_classes, activation = 'softmax'))


# Checkpointing
filepath = "/home/satyaki/Emotion/bimodal/Saved_Weights/weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

'''
# Tensorboard
tensorboardfile = "/home/satyaki/Emotion/bimodal/Logs"
tensorboard = TensorBoard(log_dir = tensorboardfile, histogram_freq = 100)
callbacks_list.append(tensorboard)

# Early Stopping
early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 50, verbose = 1, mode = 'max')
callbacks_list.append(early_stopping)
'''


# Compile the model
epochs = 1000
lrate = 0.001
decay = lrate/epochs
w_array = np.ones((4,4))
ncce = partial(w_categorical_cross_entropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
rmsprop = RMSprop(lr = lrate)
if os.path.exists(filepath):
	model.load_weights(filepath, by_name = True)
	print("Loading pre trained weights")
else:
	print("No weight file found")
model.compile(loss = 'categorical_crossentropy', optimizer = rmsprop, metrics = ['accuracy'])
print(model.summary())

def signal_handler(signal, frame):
	show_results(model, testrepo)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch = epochs, callbacks = callbacks_list, batch_size = 64, class_weight = class_weights)
show_results(model, testrepo, history)
