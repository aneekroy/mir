import os
import math
import numpy as np
import ffmpy
from python_speech_features import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import random
import Image
import csv

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
	f = open("/home/soms/EmotionMusic/MediaEval_Classification/new-label-file.txt", "w")
	infohandle = open("/home/soms/EmotionMusic/MediaEval-2013/annotations/songs_info.csv", "r")
	indexhandle = open("/home/soms/EmotionMusic/All_dist_with_songid.csv", "r")
	labelhandle = open("/home/soms/EmotionMusic/MediaEval_Classification/top_label_1to4.csv", "r")
	reader1 = csv.reader(infohandle)
	train = []
	test = []
	i = 0
	for row in reader1:
		if i !=0 :
			if row[-1] == "development":
				train.append(row[0])
			else:
				test.append(row[0])
		i += 1

	labels = {}
	indices = {}
	index = 0
	reader2 = csv.reader(indexhandle)
	for row in reader2:
		indices[index] = row[0]
		index += 1 
	
	index = 0
	reader3 = csv.reader(labelhandle)
	for row in reader3:
		labels[indices[index]] = int(row[0])
		index += 1
	
	for _,_, files in os.walk(wavdir):
		# Min shape 0 is 4458
		m = 9999
		cnt = 0
		for file in files:
			filename = file[:file.find(".wav")]	
			print(file)
			(rate, sig) = wav.read(os.path.join(wavdir, file))
			log_energy_feat = normalize(logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024))
			shape = log_energy_feat.shape
			if shape[0] < 4500:
				log_energy_feat = np.resize(np.asarray(log_energy_feat), [4500, shape[1]])
			log_energy_feat = log_energy_feat[:4500,:]
			image = np.array_split(log_energy_feat, math.ceil(log_energy_feat.shape[0]/60.))
			print("Image segmented shape", np.asarray(image).shape)
			index = 1
			if filename in train:
				cnt += 1
				for image_segment in image:
					if image_segment.shape[0] >= 57:
						image_segment = image_segment[image_segment.shape[0] - 57 : ]
					else:
						print("Shape error:",image_segment.shape[0])
					if image_segment.shape[0] < m:
						m = image_segment.shape[0]
					print(image_segment.shape)
					plt.imsave(os.path.join(featuredir + "/Train", file + "_" + str(index) + ".png"), image_segment.T, cmap = "gray")
					f.write(filename + "_" + str(index)  + " " + str(labels[filename]) + " D\n")
					index += 1
			elif filename in test:
				plt.imsave(os.path.join(featuredir + "/TestWhole", file + ".png"), log_energy_feat.T, cmap = "gray")
				cnt += 1
				for image_segment in image:
					image_segment = image_segment[image_segment.shape[0] - 57 : ]
					if image_segment.shape[0] < m:
						m = image_segment.shape[0]
					print(image_segment.shape)
					plt.imsave(os.path.join(featuredir + "/Test", file + "_" + str(index) + ".png"), image_segment.T, cmap = "gray")
					f.write(filename + "_" + str(index) + " " + str(labels[filename]) + " E\n")
					index += 1
			else:
				print("Duplicate")
			print("Count = ", str(cnt))
		print("Min shape 0:",m)	

if __name__ == "__main__":
	mp3dir = "/home/soms/EmotionMusic/MediaEval-2013/clips_45seconds"
	wavdir = "/home/soms/EmotionMusic/MediaEval/WavFiles"
	specdir = "/home/soms/EmotionMusic/Spectograms"
	featuredir = "/home/soms/EmotionMusic/MediaEval_Classification/Spec_Feat"
	#convert_mp3_to_wav(mp3dir, wavdir)
	#convert_wav_to_spec(wavdir, specdir)
	extract_features(wavdir, featuredir)
