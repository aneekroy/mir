import numpy as np
import keras
import os
import sys
import math
import signal
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Bidirectional, TimeDistributedDense, Merge, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import regularizers, GlobalAveragePooling2D, Input, Merge, merge
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
import keras.backend as K
from read_data_merge import *
import matplotlib.pyplot as plt
from itertools import product
from functools import partial
from sklearn.utils import class_weight

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
	scores = model.evaluate(xts, y_test, verbose = 1)
	print("Accuracy: %.2f%%"%(scores[1]*100))

	predictions = model.predict(xts, verbose = 1)
    #print(predictions)
	classes = np_utils.probas_to_classes(predictions)
	print(classes)
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
DATA_PATH = "/home/satyaki/Emotion/bimodal/5_Fold_Merge/Fold" + str(sys.argv[1])
LABEL_PATH = "/home/satyaki/Emotion/bimodal/annotations.csv"

x_train, y_train, x_test, y_test, testrepo, train_cnt= load_dataset(DATA_PATH, LABEL_PATH)
cw = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
print("Class Weights", cw)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

x_train = np.expand_dims(x_train, axis = 4)
x_test = np.expand_dims(x_test, axis = 4)

x_train = np.transpose(x_train, (1,0,2,3,4))
x_test = np.transpose(x_test, (1,0,2,3,4))

inputs = list()
convbranches = list()
for i in range(x_train.shape[0]):
    inputs.append(Input(shape = x_train.shape[2:]))


def base_conv(inp):
    branch = Convolution2D(32, 3, 3, border_mode = 'same', activation = 'relu', init = 'glorot_uniform')(inp)
    branch = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(branch)
    return branch

for i in range(x_train.shape[0]):
    x = base_conv(inputs[i])
    convbranches.append(x)
merged_model = merge([branch for branch in convbranches], mode = 'sum', concat_axis = -1)
merged_model = Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu', init = 'glorot_uniform')(merged_model)
merged_model = Convolution2D(128, 3, 3, border_mode = 'same', activation = 'relu', init = 'glorot_uniform')(merged_model)
merged_model = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(merged_model)
#merged_model = Convolution2D(256, 3, 3, border_mode = 'same', activation = 'relu', init = 'normal')(merged_model)
merged_model = GlobalAveragePooling2D(dim_ordering = 'tf')(merged_model)
merged_model = Dense(128, activation = 'relu', W_regularizer = l2(0.05))(merged_model)
merged_model = Dropout(0.5)(merged_model)
#merged_model = Dense(64, activation = 'relu', init = 'normal', W_regularizer = l2(0.05))(merged_model)
#merged_model = Dropout(0.5)(merged_model)
out = Dense(num_classes, activation = 'softmax')(merged_model)

model = Model(input = inputs, output = out)



# Checkpointing
dirpath = "/home/satyaki/Emotion/bimodal/Saved_Weights" + sys.argv[1]
if not os.path.exists(dirpath):
	os.makedirs(dirpath)
filepath = dirpath + "/weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]


# Tensorboard
tensorboardfile = "/home/satyaki/Emotion/bimodal/Logs"
if not os.path.exists(tensorboardfile):
    os.makedirs(tensorboardfile)
tensorboard = TensorBoard(log_dir = tensorboardfile, histogram_freq = 100)
callbacks_list.append(tensorboard)

'''
# Early Stopping
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 50, verbose = 1, mode = 'auto')
callbacks_list.append(early_stopping)
'''

# Compile the model
epochs = 500
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
ncce.__name__ = 'w_categorical_crossentropy'
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = True)
rmsprop = RMSprop()
if os.path.exists(filepath):
	model.load_weights(filepath, by_name = True)
	print("Loading pre trained weights")
else:
	print("No weight file found")
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
print("FOLD %s"%sys.argv[1])
def signal_handler(signal, frame):
	show_results(model, testrepo)
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
xtr = list()
for i in x_train:
    xtr.append(i)
xts = list()
for i in x_test:
    xts.append(i)
history = model.fit(xtr, y_train, validation_data=(xts, y_test), nb_epoch = epochs, callbacks = callbacks_list, batch_size = 16, class_weight = cw)
show_results(model, testrepo, history)
