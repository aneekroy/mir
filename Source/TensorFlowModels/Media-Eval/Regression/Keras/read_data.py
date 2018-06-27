import tensorflow as tf
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
from scipy.misc import toimage
import time
import csv

def load_dataset(data_path, label_file):
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	testrepo = []
	labels = {}
	exceptions = []
	labhandle = open(label_file, "r")
	for line in labhandle.readlines():
		line = line.split()
		labels[line[0]] = [float(line[1]), float(line[2])]
	print("Loading dataset...")
	exceptions = []
	train_path = os.path.join(data_path, "Train")
	test_path = os.path.join(data_path, "Test")
	
	for _, _, files in os.walk(train_path):
		for file in files:
			#print(file)
			filename = file[:file.find(".")]
			filename += file[file.find("_") : file.rfind(".")]
			#print(filename)
			try:
				train_y.append(labels[filename][0])
				train_x.append(np.asarray(Image.open(os.path.join(train_path, file)).convert("L")))
			except Exception as e:
				exceptions.append(e)
				
	for _, _, files in os.walk(test_path):
		for file in files:
			#print(file)
			filename = file[:file.find(".")]
			filename += file[file.find("_") : file.rfind(".")]
			#print(filename)
			try:
				test_y.append(labels[filename][0])
				test_x.append(np.asarray(Image.open(os.path.join(test_path, file)).convert("L")))
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
	return (train_x, train_y, test_x, test_y, testrepo)
	
if __name__ == "__main__":
	start_time = time.time()
	train_x, train_y, test_x, test_y, _= load_dataset("/home/soms/EmotionMusic/MediaEval/Spec_Feat", "/home/soms/EmotionMusic/MediaEval/new-label-file.txt")
	duration = time.time() - start_time
	print("It took %.3f sec to load the dataset"%duration)
	
