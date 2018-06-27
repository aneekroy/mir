import numpy as np
import keras
import os
import sys
import math
import signal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Bidirectional, TimeDistributedDense, Merge, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import regularizers, GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
import keras.backend as K
from read_data import *
import matplotlib.pyplot as plt
from itertools import product
from functools import partial
from sklearn.utils import class_weight
import post_process

MA = 0
best_epoch = 0

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
		global MA, best_epoch
		model, testrepo = self.test_data
		show_results(model, testrepo)
		acc = post_process.get_accuracy("/home/satyaki/Emotion/Soundtrack/sequence.txt")
		if acc> MA:
			MA = acc
			best_epoch = epoch + 1

		
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
	conf_mat = np.zeros((4,4), dtype = np.float32)
	predictions = model.predict_classes(x_test, verbose = 1)
	index = 0
	#for p, a in zip(predictions, y_test):
	#	print("File_Segment:" + testrepo[index] + " Actual:" + str(np.argmax(a)) + " Predicted:" + str(p))
	#	index += 1
	segment_indices = {}
	for index in range(len(testrepo)):
		file = testrepo[index]
		filename = file[:file.find("_")]
		fileindex = file[file.find("_") + 1 : file.find(".")]
		if filename in segment_indices:
			segment_indices[filename].append([fileindex, index])
		else:
			segment_indices[filename] = [[fileindex, index]]
	print(len(segment_indices.keys()))
	print("Prediction length:%d Actual_length:%d FileRepo_length:%d"%(len(predictions), len(y_test), len(testrepo)))
	f = open("/home/satyaki/Emotion/Soundtrack/sequence.txt", "w")
	whole_set_accuracy = 0.0
	LAB_MAP = {0:"T", 1:"S", 2:"A/F", 3:"H"}
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
		conf_mat[predicted, actual] += 1
		if predicted == actual:
			whole_set_accuracy += 1
	whole_set_accuracy = float(whole_set_accuracy) / len(segment_indices.keys())
	print("Whole set accuracy:%.3f"%whole_set_accuracy)
	s = np.sum(conf_mat)
	for i in range(num_classes):
		for j in range(num_classes):
			conf_mat[i,j] /= s
			conf_mat[i,j] *= 100
			print "%.2f"%conf_mat[i,j],
		print("\n")

	for i in range(len(conf_mat)):
		act = conf_mat[i,i]
		s1 = 0
		s2 = 0
		for j in range(len(conf_mat[i])):
				s1 += conf_mat[i,j]
				s2 += conf_mat[j,i]
		if s1 != 0:
			pr = float(act)/s1
		else:
			pr = 0
		if s2 != 0:
			re = float(act)/s2
		else:
			re = 0
		if pr != 0 and re != 0:
			f1 = float(2 * pr * re)/(pr + re)
		else:
			f1 = 0
		print("%d Precision:%.2f Recall:%.2f F-Measure:%.2f"%(i, pr, re, f1))
	
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


def get_weights(lab_dict):
	S = np.sum(lab_dict.values())
	cw = {}
	print("Class weights")
	for key in lab_dict.keys():
		cw[key] = float(lab_dict[key]) / S
		print(str(key) + ":" + str(cw[key]))
	return cw
'''
def class_weights(lab_dict, mu = 0.15):
	total = sum(lab_dict.values())
	keys = lab_dict.keys()
	class_weight = {}
	for key in keys:
		score = math.log(mu*total/float(lab_dict[key]))
		class_weight[key] = score if score > 1.0 else 1.0
		print(str(key) + ":" + str(class_weight[key]))
	return class_weight
'''

seed = 7
np.random.seed(7)
DATA_PATH = "/home/satyaki/Emotion/Soundtrack/5_Fold/Fold" + str(sys.argv[1])
LABEL_PATH = "/home/satyaki/Emotion/Soundtrack/labels.txt"

x_train, y_train, x_test, y_test, testrepo, train_cnt, test_cnt= load_dataset(DATA_PATH, LABEL_PATH)
#cw = class_weights(train_cnt)
cw = class_weight.compute_class_weight('balanced', np.unique(y_test), y_test)
print("Class Weight", cw)
weights = get_weights(test_cnt)
print("Weights", weights)
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

# ConvConvPool1
model.add(Convolution2D(64, 3, 3, input_shape = (x_train.shape[1:]), border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(Convolution2D(64, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#model.add(Dropout(0.35))

# ConvConvPool2
model.add(Convolution2D(64, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(Convolution2D(64, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#model.add(Dropout(0.35))

# ConvPool3
model.add(Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
#model.add(Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Dropout(0.35))

# ConvPool4
model.add(Convolution2D(256, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
#model.add(Convolution2D(256, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size = (3, 3), strides = (3, 3)))

model.add(Dropout(0.35))


# ConvConv4
model.add(Convolution2D(256, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
#model.add(Convolution2D(256, 3, 3, border_mode = 'same', activation = 'relu', W_constraint = maxnorm(3), init = 'glorot_uniform', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size = (3, 3), strides = (3, 3)))

model.add(Dropout(0.35))

# Global Average Pooling
#model.add(GlobalAveragePooling2D(dim_ordering = 'tf'))


# Flatten
model.add(Flatten())

# Dense1
model.add(Dense(256, init = 'uniform', activation = 'relu', W_regularizer = l2(0.001)))

# Dropout3
model.add(Dropout(0.35))

# Dense2
model.add(Dense(256, init = 'uniform', activation = 'relu', W_regularizer = l2(0.001)))

# Dropout4
model.add(Dropout(0.35))


# Softmax
model.add(Dense(num_classes, activation = 'softmax'))




# Compile the model
epochs = 100
lrate = 0.01
decay = lrate/epochs
w_array = np.zeros((4,4))
w_array[0,1] = max(1, cw[0]/cw[1])
w_array[0,2] = max(1, cw[0]/cw[2])
w_array[0,3] = max(1, cw[0]/cw[3])
w_array[1,0] = max(1, cw[1]/cw[0])
w_array[1,2] = max(1, cw[1]/cw[2])
w_array[1,3] = max(1, cw[1]/cw[3])
w_array[2,0] = max(1, cw[2]/cw[0])
w_array[2,1] = max(1, cw[2]/cw[1])
w_array[2,3] = max(1, cw[2]/cw[3])
w_array[3,0] = max(1, cw[3]/cw[0])
w_array[3,1] = max(1, cw[3]/cw[1])
w_array[3,2] = max(1, cw[3]/cw[2])
print w_array
ncce = partial(w_categorical_cross_entropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'
sgd = SGD(lr = lrate, momentum = 0.5, decay = decay, nesterov = False)
rmsprop = RMSprop(lr = lrate, decay = decay)
adadelta = Adadelta(lr = lrate, decay = decay)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Checkpointing
dirpath = "/home/satyaki/Emotion/Soundtrack/Saved_Weights" + sys.argv[1]
if not os.path.exists(dirpath):
	os.makedirs(dirpath)
filepath = dirpath + "/weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]
callbacks_list.append(TestCallback((model, testrepo)))
'''
# Tensorboard
tensorboardfile = "/home/soms/EmotionMusic/bimodal/Logs"
tensorboard = TensorBoard(log_dir = tensorboardfile, histogram_freq = 100)
callbacks_list.append(tensorboard)


# Early Stopping
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 5, verbose = 1, mode = 'auto')
callbacks_list.append(early_stopping)
'''

if os.path.exists(filepath):
	model.load_weights(filepath, by_name = True)
	print("Loading pre trained weights")
else:
	print("No weight file found")
print(model.summary())
print("FOLD %s"%sys.argv[1])
def signal_handler(signal, frame):
	show_results(model, testrepo)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
f = open("res.txt", "a")
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch = epochs, shuffle = False, callbacks = callbacks_list, batch_size = 128, class_weight = cw)
show_results(model, testrepo, history)
print("Best accuracy %.4f obtained at epoch %d"%(MA, best_epoch))
