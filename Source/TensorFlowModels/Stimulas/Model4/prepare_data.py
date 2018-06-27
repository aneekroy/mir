import os
import math
import numpy as np
import ffmpy
from python_speech_features import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import random
import Image

# This is subject to change if cross-validation is applied. Sticking to this cheap stuff for the time being.
TRAIN_PERCENT = 0.8

def normalize(A):
	M = [[x]*A.shape[1] for x in np.mean(A, axis = 1)]
	S = [[x]*A.shape[1] for x in np.std(A, axis = 1) if x is not 0]
	A = (A - M)
	for i in range(len(A)):
		for j in range(len(A[i])):
			if S[i][j] != 0.0:
				A[i][j] = A[i][j]/S[i][j]
	return A

LAB_CLASSES = {"Happy":0, "Sad":1, "Tender":2, "Anger_Fear":3}


def extract_features(wavdir, featuredir):
	f = open("/home/soms/EmotionMusic/new-label-file.txt", "w")
	logger = open("info.txt", "w")
	m = 99999
	for _, classes, _ in os.walk(wavdir):
		for label in classes:
			labdir = os.path.join(wavdir, label)
			for _,_, files in os.walk(labdir):	
				trainset = files[:int(TRAIN_PERCENT*len(files))]
				testset = files[int(TRAIN_PERCENT*len(files)):]
				classindex = LAB_CLASSES[label]
				for file in trainset:
					print("File %s"%file)
					filename = file[:file.find(".wav")]
					logger.write(file + " train\n")
					(rate, sig) = wav.read(os.path.join(labdir, file))
					if np.ndim(sig) == 2:
						sig = np.mean(sig, axis=1)
					sz = sig.shape[0] / rate
					sig = sig[:sz*rate]
					sig = np.split(sig, sz)
					index = 0
					for sig_seg in sig:
						log_energy_feat = normalize(logfbank(sig_seg, rate, winlen = 0.025, winstep = 0.01, nfilt = 128, nfft = 1024))
						print(log_energy_feat.shape)
						plt.imsave(os.path.join(featuredir + "/Train", str(classindex) + "_" + filename + "_" + str(index) + ".png"), log_energy_feat.T, cmap = "gray")
						#f.write(os.path.join(featuredir + "/Train", str(classindex) + "_" + filename + "_" + str(index) + ".png") + ":" + str(classindex)+"\n")
						index += 1
				for file in testset:
					fileindex = int(file[:file.find(".wav")])
					print("File %s"%file)
					filename = file[:file.find(".wav")]
					logger.write(file + " test\n")
					(rate, sig) = wav.read(os.path.join(labdir, file))
					sig = sig[:, 0]
					log_energy_feat = normalize(logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 128, nfft = 1024))
					if np.ndim(sig) == 2:
						sig = np.mean(sig, axis=1)
					sz = sig.shape[0] / rate
					sig = sig[:sz*rate]
					sig = np.split(sig, sz)
					index = 0
					for sig_seg in sig:
						log_energy_feat = normalize(logfbank(sig_seg, rate, winlen = 0.025, winstep = 0.01, nfilt = 128, nfft = 1024))
						print(log_energy_feat.shape)
						plt.imsave(os.path.join(featuredir + "/Test", str(classindex) + "_" + filename + "_" + str(index) + ".png"), log_energy_feat.T, cmap = "gray")
						#f.write(os.path.join(featuredir + "/Train", str(classindex) + "_" + filename + "_" + str(index) + ".png") + ":" + str(classindex)+"\n")
						index += 1
			print(m)			

if __name__ == "__main__":
	wavdir = "/home/soms/EmotionMusic/WavFiles"
	featuredir = "/home/soms/EmotionMusic/New_Segmented_Features"
	extract_features(wavdir, featuredir)
