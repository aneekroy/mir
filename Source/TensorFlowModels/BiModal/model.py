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
	scores = model.evaluate(x_test, y_test, verbose = 1)
	print("Accuracy: %.2f%%"%(scores[1]*100))

	predictions = model.predict_classes(x_test, verbose = 1)
	index = 0
	for p, a in zip(predictions, y_test):
		print("File_Segment:" + testrepo[index] + " Actual:" + str(np.argmax(a)) + " Predicted:" + str(p))
		index += 1
	segment_indices = {}
	for index in range(len(testrepo)):
		file = testrepo[index]
		filename = file[:file.find("_")]
		if "-" in filename:
			filename = filename.split("-")[0]
		fileindex = file[file.find("_") + 1 : file.find(".")]
		if filename in segment_indices:
			segment_indices[filename].append([fileindex, index])
		else:
			segment_indices[filename] = [[fileindex, index]]
	print(len(segment_indices.keys()))
	print("Prediction length:%d Actual_length:%d FileRepo_length:%d"%(len(predictions), len(y_test), len(testrepo)))
	f = open("/home/soms/EmotionMusic/bimodal/sequence.txt", "w")
	whole_set_accuracy = 0.0
	LAB_MAP = {0:"A", 1:"B", 2:"C", 3:"D"}
	for s in segment_indices:
		cnt = {}
		indices = segment_indices[s] 
		actual = np.argmax(y_test[indices[0][1]])
		for index in indices:
			if predictions[index[1]] in cnt:
				cnt[predictions[index[1]]] += 1
			else:
				cnt[predictions[index[1]]] = 1
		for i in range(len(indices)):
			indices[i][0] = int(indices[i][0])
		sorted_indices = sorted(indices)
		for index in sorted_indices:
			f.write(LAB_MAP[predictions[index[1]]] + " ")
		f.write(str(actual) + "\n")
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
		
		
def class_weights(lab_dict):
	M = max(lab_dict.values())
	cw = {}
	print("Class weights")
	for key in lab_dict.keys():
		cw[key] = float(M) / lab_dict[key]
		print(str(key) + ":" + str(cw[key]))
	return cw

seed = 7
np.random.seed(7)
DATA_PATH = "/home/soms/EmotionMusic/bimodal/Features"
LABEL_PATH = "/home/soms/EmotionMusic/MER_bimodal_dataset/annotations.csv"

x_train, y_train, x_test, y_test, testrepo, train_cnt= load_dataset(DATA_PATH, LABEL_PATH)
cw = class_weights(train_cnt)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)

model = Sequential()

# Gaussian Noise
model.add(GaussianNoise(input_shape = x_train.shape[1:], sigma = 0.5))

# ConvPool1
model.add(Convolution2D(60, 4, 4, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size = (3, 3), strides = (3, 3)))

# Conv2
model.add(Convolution2D(70, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(BatchNormalization(axis = 3))

# Conv3
model.add(Convolution2D(80, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3)))
model.add(BatchNormalization(axis = 3))

# GlobalAvgPool
model.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))

# Reshape
model.add(Reshape((9, 5*80)))

'''
# Bidirectional LSTM 1
model.add(Bidirectional(LSTM(output_dim = 128, init = 'uniform', inner_init = 'uniform', forget_bias_init = 'uniform', return_sequences = True, activation = 'relu', inner_activation = 'hard_sigmoid',W_regularizer = l2(0.1), U_regularizer = l2(0.1), dropout_W = 0.5, dropout_U = 0.5)))
'''

# Bidirectional LSTM1
model.add(Bidirectional(LSTM(output_dim = 128, return_sequences = False)))

# Dropout
model.add(Dropout(0.5))

# Softmax
model.add(Dense(num_classes, activation = 'softmax'))


# Checkpointing
filepath = "/home/soms/EmotionMusic/bimodal/Saved_Weights/weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
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
w_array = np.ones((4,4))
w_array[0,1] = 1.2
w_array[0,2] = 1.2
w_array[0,3] = 1.2	
w_array[1,0] = 1.5
w_array[1,2] = 1.5
w_array[1,3] = 1.5
w_array[2,0] = 2.5
w_array[2,1] = 2.5
w_array[2,3] = 2.5
w_array[3,0] = 1.5
w_array[3,1] = 1.5
w_array[3,2] = 1.5
ncce = partial(w_categorical_cross_entropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = True)
rmsprop = RMSprop()
if os.path.exists(filepath):
	model.load_weights(filepath, by_name = True)
	print("Loading pre trained weights")
else:
	print("No weight file found")
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

def signal_handler(signal, frame):
	show_results(model, testrepo)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

history = model.fit(x_train, y_train, validation_data=(x_train, y_train), nb_epoch = epochs, callbacks = callbacks_list, batch_size = 32)
show_results(model, testrepo, history)
