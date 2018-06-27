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

'''
Converts mp3 files in mp3dir to wav files in wavdir
'''
def convert_mp3_to_wav(mp3dir, wavdir):
	for _,_,files in os.walk(mp3dir):
		for file in files:
			filename = file[:file.find(".mp3")]
			ff = ffmpy.FFmpeg(inputs = {os.path.join(mp3dir, filename + ".mp3"):None}, outputs = {os.path.join(wavdir, filename + ".wav"):None})
			ff.run()

'''
Converts wav files in wavdir to corresponding spectograms in specdir 
'''
def convert_wav_to_spec(wavdir, specdir):
	for _,_, files in os.walk(wavdir):
		for file in files:
			filename = file[:file.find(".wav")]
			(rate, sig) = wav.read(os.path.join(wavdir, file))
			sig = sig[:, 0]
			plt.clf()
			_,_,_,spec = plt.specgram(sig, NFFT = 256, Fs = rate, noverlap = 16, cmap = plt.cm.jet)
			image = Image.fromarray(spec)
			image = image.convert("L")
			image.save(os.path.join(specdir, filename + ".jpg"))

LAB_CLASSES = {"Happy":0, "Sad":1, "Tender":2, "Anger_Fear":3}

'''
Extracts mfcc and delta features from the wav files and 
'''
def extract_features(wavdir, featuredir):
	f = open("/home/soms/EmotionMusic/new-label-file.txt", "w")
	logger = open("info.txt", "w")
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
					sig = sig[:, 0]
					log_energy_feat = normalize(logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024))
					image = np.array_split(log_energy_feat, math.ceil(log_energy_feat.shape[0]/60.))
					index = 1
					for image_segment in image:
						image_segment = image_segment[image_segment.shape[0] - 57 : ]
						#print(image_segment.T.shape)
						plt.imsave(os.path.join(featuredir + "/Train", str(classindex) + "_" + filename + "_" + str(index) + ".png"), image_segment.T, cmap = "gray")
						f.write(os.path.join(featuredir + "/Train", str(classindex) + "_" + filename + "_" + str(index) + ".png") + ":" + str(classindex)+"\n")
						index += 1
				for file in testset:
					fileindex = int(file[:file.find(".wav")])
					print("File %s"%file)
					filename = file[:file.find(".wav")]
					logger.write(file + " test\n")
					(rate, sig) = wav.read(os.path.join(labdir, file))
					sig = sig[:, 0]
					log_energy_feat = normalize(logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024))
					plt.imsave(os.path.join(featuredir + "/Test", str(classindex) + "_" + filename + ".png"), log_energy_feat, cmap = "gray")
					f.write(os.path.join(featuredir + "/Test", str(classindex) + "_" + filename +  ".png") + ":" + str(classindex)+"\n")
					index += 1
				
if __name__ == "__main__":
	mp3dir = "/home/soms/EmotionMusic/emotion/emotion-train"
	wavdir = "/home/soms/EmotionMusic/WavFiles"
	specdir = "/home/soms/EmotionMusic/Spectograms"
	featuredir = "/home/soms/EmotionMusic/Spec_Feat"
	#convert_mp3_to_wav(mp3dir, wavdir)
	#convert_wav_to_spec(wavdir, specdir)
	extract_features(wavdir, featuredir)
