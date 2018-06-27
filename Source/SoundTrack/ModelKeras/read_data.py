import tensorflow as tf
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
from scipy.misc import toimage
import time
import csv

def load_dataset(data_path, label_file):
	start_time = time.time()
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	testrepo = []
	print("Loading dataset...")
	trainC = {}
	testC = {}
	labels = {}
	labdict = {}
	invdict = {}
	labindex = 0
	train_path = os.path.join(data_path, "Train")
	test_path = os.path.join(data_path, "Test")
	exceptions = []

	labhandle = open(label_file, "r")

	for line in labhandle.readlines():
		line = line.split(' ')
        	if line[1] not in labdict:
            		labdict[line[1]] = labindex
			invdict[labindex] = line[1]
            		labindex += 1
		labels[line[0]] = labdict[line[1]]

	for _, _, files in os.walk(train_path):
		for file in files:
			filename = file[:file.find(".")]
			#print filename
			train_y.append(labels[filename])
			train_x.append(np.asarray(Image.open(os.path.join(train_path, file)).convert("L")))
			if labels[filename] in trainC:
				trainC[labels[filename]] += 1
			else:
				trainC[labels[filename]] = 1


	for _, _, files in os.walk(test_path):
		for file in files:
			filename = file[:file.find(".")]
			index = filename[filename.rfind("_") + 1:]
			test_y.append(labels[filename])
			test_x.append(np.asarray(Image.open(os.path.join(test_path, file)).convert("L")))
			if labels[filename] in testC:
				testC[labels[filename]] += 1
			else:
				testC[labels[filename]] = 1
			testrepo.append(file)	


	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)
	print("TrainX:",train_x.shape)
	print("TrainY:",train_y.shape)
	print("TestX:",test_x.shape)
	print("TestY:",test_y.shape)
	print(invdict)
	print(testrepo)
	print("Loading complete...")
	s_train = 0
	s_test = 0
	for x in trainC:
		s_train += trainC[x]
	for x in testC:
		s_test += testC[x]
	print("train")
	for x in trainC:
		print(str(x) + ":" + str(float(trainC[x])/s_train))
	print("test")
	for x in testC:
		print(str(x) + ":" + str(float(testC[x])/s_test))
	duration = time.time() - start_time
	print("It took %.3f sec to load the dataset"%duration)
	return (train_x, train_y, test_x, test_y, testrepo, trainC, testC)

if __name__ == "__main__":
	train_x, train_y, test_x, test_y, _, _, _= load_dataset("/home/satyaki/Emotion/Soundtrack/5_Fold/Fold0", "/home/satyaki/Emotion/Soundtrack/labels.txt")
