import tensorflow as tf
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
from scipy.misc import toimage
import time

def load_dataset(data_path):
	start_time = time.time()
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	testrepo = []
	trainC = {}
	print("Loading dataset...")
	train_path = os.path.join(data_path, "Train")
	test_path = os.path.join(data_path, "Test")
	for _, _, files in os.walk(train_path):
		for file in files:
			#print(file)
			train_y.append(int(file[:file.find("_")]))
			train_x.append(np.asarray(Image.open(os.path.join(train_path, file)).convert("L")))
			if file[:file.find("_")] in trainC:
				trainC[file[:file.find("_")]] += 1
			else:
				trainC[file[:file.find("_")]] = 1
	for _, _, files in os.walk(test_path):
		for file in files:
			#print(file)
			test_y.append(int(file[:file.find("_")]))
			test_x.append(np.asarray(Image.open(os.path.join(test_path, file)).convert("L")))
			testrepo.append(file)
	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)
	print("TrainX:",train_x.shape)
	print("TrainY:",train_y.shape)
	print("TestX:",test_x.shape)
	print("TestY:",test_y.shape)
	print("Loading complete...")
	duration = time.time() - start_time
	print("It took %.3f s to load the dataset"%duration)
	return (train_x, train_y, test_x, test_y, testrepo, trainC)
	
if __name__ == "__main__":
	train_x, train_y, test_x, test_y, _, _ = load_dataset("/home/soms/EmotionMusic/Segmented_Features")
	for i in range(9):
		plt.subplot(330 + 1 + i)
		plt.imshow(toimage(train_x[i]))
		
	plt.show()
	
