import os
import tensorflow as tf
import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt

data_path = "/home/soms/EmotionMusic/Hindi_Raga_Spec"
image_path = "/home/soms/EmotionMusic/Spec_Images"

for _,dirs, _ in os.walk(data_path):
	for dir in dirs:
		os.makedirs(image_path + "/" + dir)
		for _,_,files in os.walk(data_path + "/" + dir):
			for file in files:
				image_array = np.genfromtxt(data_path + "/" + dir + "/" + file, delimiter = ",")
				#scipy.misc.imsave(image_path + "/" + dir + "/" + file[:file.find(".txt")] + ".jpg", image_array)
				scipy.misc.toimage(image_array, cmax = 1.0, cmin = 0.0).save(image_path + "/" + dir + "/" + file[:file.find(".txt")] + ".jpg")
		print("Dir %s conversion complete."%dir)