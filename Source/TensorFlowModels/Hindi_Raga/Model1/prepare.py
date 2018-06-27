import os
import sys
from python_speech_features import *
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PERCENT = 0.8

'''
zero mean and unit std
'''
def normalize(A):
	M = [[x]*A.shape[1] for x in np.mean(A, axis = 1)]
	S = [[x]*A.shape[1] for x in np.std(A, axis = 1) if x is not 0]
	A = (A - M)
	for i in range(len(A)):
		for j in range(len(A[i])):
			if S[i][j] != 0.0:
				A[i][j] = A[i][j]/S[i][j]
	return A


'''
Extracts spectograms for the wav files 
'''
def extract_features(wavdir, featuredir):
	labelfile = open("/home/soms/EmotionMusic/ModelRaga/new-label-file.txt", "w")
	for _, classes, _ in os.walk(wavdir):
		for lab_index, label in enumerate(classes):
			labdir = os.path.join(wavdir, label)
			print(labdir)
			for _,_, files in os.walk(labdir):	
				trainset = files[:int(TRAIN_PERCENT*len(files))]
				testset = files[int(TRAIN_PERCENT*len(files)):]
				#classindex = LAB_CLASSES[label]
				for file in trainset:
					#print("File %s"%file)
					filename = file[:file.find(".wav")]
					(rate, sig) = wav.read(os.path.join(labdir, file))
					log_energy_feat = normalize(logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 64, nfft = 1024))
					plt.imsave(os.path.join(featuredir + "/Train", filename + ".jpg"), log_energy_feat.T, cmap = "gray")
					labelfile.write(filename + ":-" + str(lab_index) + "\n")
				for file in testset:
					#print("File %s"%file)
					filename = file[:file.find(".wav")]
					(rate, sig) = wav.read(os.path.join(labdir, file))
					log_energy_feat = normalize(logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 64, nfft = 1024))
					plt.imsave(os.path.join(featuredir + "/Test", filename + ".jpg"), log_energy_feat.T, cmap = "gray")
					labelfile.write(filename + ":-" + str(lab_index) + "\n")

if __name__ == "__main__":
	extract_features("/home/soms/EmotionMusic/Hindi_Raga_Clips", "/home/soms/EmotionMusic/ModelRaga/Spectograms")