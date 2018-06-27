import tensorflow as tf
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
from scipy.misc import toimage
import time
import csv
from sklearn.model_selection import StratifiedKFold

def load_dataset(data_path, label_file):
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	data_x = []
	data_y = []
	testrepo = []
	labhandle = open(label_file, "r")
	LABS = {"Q1":0, "Q2":1, "Q3":2, "Q4":3}
	reader = csv.reader(labhandle)
	labels = {}
	i = 0
	flag = False
	dup = "0"
	for row in reader:
		if i != 0:
			if row[2] in labels:
				flag = True
				dup = row[2]
			labels[row[2]] = LABS[row[3]]
			#print(row[2] + " " + row[3])
		i += 1
	if flag == True:
		print("Duplicate present", dup)

	print("Loading dataset...")
	exceptions = []
	trainC = {}
	testC = {}
	train_path = os.path.join(data_path, "Train")
	test_path = os.path.join(data_path, "Test")
	#print("="*10)
	#print("Train")
	#print("="*10)
	for _, _, files in os.walk(train_path):
		for file in files:
		#	print(file)
			filename = file[:file.find("_")]
			if "-" in filename:
				filename = filename.split("-")[0]
			try:
				train_y.append(labels[filename])
				data = np.asarray(Image.open(os.path.join(train_path, file)).convert("L")).T
				data = data[:196,:]
				train_x.append(data)
				data_y.append(labels[filename])
				data_x.append(data)
				if labels[filename] in trainC:
					trainC[labels[filename]] += 1
				else:
					trainC[labels[filename]] = 1
			except Exception as e:
				exceptions.append(e)

	#print("="*10)
	#print("Test")
	#print("="*10)
	for _, _, files in os.walk(test_path):
		for file in files:
		#	print(file)
			filename = file[:file.find("_")]
			if "-" in filename:
				filename = filename.split("-")[0]
			try:
				test_y.append(labels[filename])
				data = np.asarray(Image.open(os.path.join(test_path, file)).convert("L")).T
				data = data[:196,:]
				test_x.append(data)
				data_y.append(labels[filename])
				data_x.append(data)
				if labels[filename] in testC:
					testC[labels[filename]] += 1
				else:
					testC[labels[filename]] = 1
				testrepo.append(file)
			except Exception as e:
				exceptions.append(e)
	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)
	print("TrainX:",train_x.shape)
	print("TrainY:",train_y.shape)
	print("TestX:",test_x.shape)
	print("TestY:",test_y.shape)
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
	return (train_x, train_y, test_x, test_y, testrepo, trainC, testC)
	#skf = StratifiedKFold(n_splits = 5)
	#index = 0
	#for train, test in skf.split(data_x, data_y):




if __name__ == "__main__":
	start_time = time.time()
	train_x, train_y, test_x, test_y, _, _= load_dataset("/home/satyaki/Emotion/bimodal/5_Fold_En/Fold0", "/home/satyaki/Emotion/bimodal/annotations.csv")
	duration = time.time() - start_time
	print("It took %.3f sec to load the dataset"%duration)
