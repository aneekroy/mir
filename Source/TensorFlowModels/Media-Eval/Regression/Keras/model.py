import numpy as np
import keras
import os
import sys
import signal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Bidirectional, TimeDistributedDense, Merge, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
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
	#scores = model.evaluate(x_test, y_test, verbose = 1)
	#print("Score: %.2f%%"%(scores[1]*100))
	f = open("/home/soms/EmotionMusic/MediaEval/outputs/predictions.txt", "w")
	predictions = model.predict(x_test, verbose = 1)
	index = 0
	for p, a in zip(predictions, y_test):
		print(testrepo[index] + " " + str(a) + " " + str(p[0]))
		f.write(testrepo[index] + " " + str(a) + " " + str(p[0]) + "\n")
		index += 1
	
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
		

seed = 7
np.random.seed(7)
DATA_PATH = "/home/soms/EmotionMusic/MediaEval/Spec_Feat"
LABEL_PATH = "/home/soms/EmotionMusic/MediaEval/new-label-file.txt"

x_train, y_train, x_test, y_test, testrepo= load_dataset(DATA_PATH, LABEL_PATH)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)

model = Sequential()

# ConvPool1
model.add(Convolution2D(64, 3, 3, input_shape = (x_train.shape[1:]), border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# ConvPool2
model.add(Convolution2D(64, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# ConvPool3
model.add(Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Flatten
model.add(Flatten())

# Dense1
model.add(Dense(128, activation = 'relu', W_regularizer = l2(0.01)))

# Dropout1
model.add(Dropout(0.5))

# Dense2
model.add(Dense(64, activation = 'relu', W_regularizer = l2(0.01)))

# Dropout1
model.add(Dropout(0.5))


# Softmax
model.add(Dense(1, activation = 'linear'))


# Checkpointing
filepath = "/home/soms/EmotionMusic/MediaEval/Saved_Weights/weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
callbacks_list = [checkpoint]

'''
# Tensorboard
tensorboardfile = "/home/soms/EmotionMusic/bimodal/Logs"
tensorboard = TensorBoard(log_dir = tensorboardfile, histogram_freq = 100)
callbacks_list.append(tensorboard)

# Early Stopping
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 5, verbose = 1, mode = 'auto')
callbacks_list.append(early_stopping)
'''

# Compile the model
epochs = 500
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = True)
if os.path.exists(filepath):
	model.load_weights(filepath, by_name = True)
	print("Loading pre trained weights")
else:
	print("No weight file found")
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
print(model.summary())

def signal_handler(signal, frame):
	show_results(model, testrepo)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch = epochs, callbacks = callbacks_list, batch_size = 32)
show_results(model, testrepo, history)
