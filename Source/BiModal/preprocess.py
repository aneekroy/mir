import os
import ffmpy
from python_speech_features import *
import scipy.io.wavfile as wav
import numpy as np
import math
import matplotlib.pyplot as plt

TRAIN_PERCENT = 0.8

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

def convert_mp3_to_wav(mp3dir, wavdir):
	for _,_,files in os.walk(mp3dir):
		for file in files:
			filename = file[:file.find(".mp3")]
			ff = ffmpy.FFmpeg(inputs = {os.path.join(mp3dir, filename + ".mp3"):None}, outputs = {os.path.join(wavdir, filename + ".wav"):None})
			ff.run()


def extract_features(wavdir, featuredir):
	f = open("/home/soms/EmotionMusic/bimodal/new-label-file.txt", "w")
	#logger = open("info.txt", "w")
	m = 9999
	for _,_, files in os.walk(wavdir):	
		trainset = files[:int(TRAIN_PERCENT*len(files))]
		testset = files[int(TRAIN_PERCENT*len(files)):]
		traincnt = 0
		testcnt = 0
		for file in trainset:
			traincnt += 1
			print("File %s"%file)
			filename = file[:file.find(".wav")]
			(rate, sig) = wav.read(os.path.join(wavdir, file))
			if np.ndim(sig) == 2:
				sig = np.mean(sig, 1)
			log_energy_feat = logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024)
			image = np.array_split(log_energy_feat, math.ceil(log_energy_feat.shape[0]/60.))
			index = 1
			for image_segment in image:
				image_segment = image_segment[image_segment.shape[0] - 58 : ]
				print(image_segment.T.shape)
				plt.imsave(os.path.join(featuredir + "/Train", filename + "_" + str(index) + ".png"), image_segment, cmap = "gray")
				f.write(os.path.join(featuredir + "/Train", filename + "_" + str(index) + ".png") + " train\n")
				index += 1
		for file in testset:
			testcnt += 1
			print("File %s"%file)
			filename = file[:file.find(".wav")]
			(rate, sig) = wav.read(os.path.join(wavdir, file))
			if np.ndim(sig) == 2:
				sig = np.mean(sig, 1)
			log_energy_feat = logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024)
			image = np.array_split(log_energy_feat, math.ceil(log_energy_feat.shape[0]/60.))
			index = 1
			for image_segment in image:
				image_segment = image_segment[image_segment.shape[0] - 58 : ]
				print(image_segment.T.shape)
				plt.imsave(os.path.join(featuredir + "/Test", filename + "_" + str(index) + ".png"), image_segment, cmap = "gray")
				f.write(os.path.join(featuredir + "/Test", filename + "_" + str(index) + ".png") + " test\n")
				index += 1
		print("Traincount:", traincnt)
		print("Testcount:", testcnt)
if __name__ == "__main__":
	#convert_mp3_to_wav("/home/soms/EmotionMusic/MER_bimodal_dataset/Corpus-Audio-162", "/home/soms/EmotionMusic/MER_bimodal_dataset/Wavs")
	extract_features("/home/soms/EmotionMusic/bimodal/Wavs", "/home/soms/EmotionMusic/bimodal/Features")
