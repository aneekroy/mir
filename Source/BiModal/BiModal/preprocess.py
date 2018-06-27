import os
import csv
import ffmpy
from python_speech_features import *
import scipy.io.wavfile as wav
import numpy as np
import math
import matplotlib.pyplot as plt
import random

TRAIN_PERCENT = 0.7

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

def data_augmentation():
	pass


def write_features(file, wavdir, outdir):
	filename = file[:file.find(".wav")]
	(rate, sig) = wav.read(os.path.join(wavdir, file))
	if np.ndim(sig) == 2:
		sig = np.mean(sig, 1)
	log_energy_feat = logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024)
	image = np.array_split(log_energy_feat, math.ceil(log_energy_feat.shape[0]/60.))
	index = 1
	for image_segment in image:
		image_segment = image_segment[image_segment.shape[0] - 58 : ]
		#print(image_segment.T.shape)
		plt.imsave(os.path.join(outdir, filename + "_" + str(index) + ".png"), image_segment, cmap = "gray")
		index += 1

def write_record(file, wavdir, outdir):
	filename = file[:file.find(".wav")]
	(rate, sig) = wav.read(os.path.join(wavdir, file))
	if np.ndim(sig) == 2:
		sig = np.mean(sig, 1)
	log_energy_feat = logfbank(sig, rate, winlen = 0.025, winstep = 0.01, nfilt = 32, nfft = 1024)
	plt.imsave(os.path.join(outdir, filename + ".png"), log_energy_feat, cmap = "gray")


def create_k_fold(wavdir, featuredir, labelfile, k = 10):
	f = open("/home/satyaki/Emotion/bimodal/new-label-file.txt", "w")
	random.seed(7)
	data = None
	for _,_, files in os.walk(wavdir):
		data = files
	labhandle = open(labelfile, "r")
	LABS = {"Q1":0, "Q2":1, "Q3":2, "Q4":3}
	reader = csv.reader(labhandle)
	labels = {}
	i = 0
	flag = False
	dup = "0"
	sampledict = {}
	for row in reader:
		if i != 0:
			if row[2] in labels:
				flag = True
				dup = row[2]
			labels[row[2]] = LABS[row[3]]
			#print(row[2] + " " + row[3])
		i += 1
	for sample in data:
		if sample == "A008.wav":
			data.remove(sample)
			continue
		filename = sample[:sample.find(".")]
		if "-" in filename:
			filename = filename.split("-")[0]
		if labels[filename] in sampledict:
			sampledict[labels[filename]].append(sample)
		else:
			sampledict[labels[filename]] = [sample]
	stratified_samples = {}
	for label in sampledict:
		random.shuffle(sampledict[label])
		stratified_samples[label] = np.array_split(sampledict[label], k)
	for index in range(k):
		print("Creating fold %d"%index)
		os.makedirs(featuredir + "/Fold" + str(index))
		os.makedirs(featuredir + "/Fold" + str(index) + "/Train")
		os.makedirs(featuredir + "/Fold" + str(index) + "/Test")
		for label in stratified_samples:
			for i in range(len(stratified_samples[label])):
				if i != index:
					for file in stratified_samples[label][i]:
						write_record(file, wavdir, featuredir + "/Fold" + str(index) + "/Train")
				else:
					for file in stratified_samples[label][i]:
						write_record(file, wavdir, featuredir + "/Fold" + str(index) + "/Test")



if __name__ == "__main__":
	#convert_mp3_to_wav("/home/soms/EmotionMusic/MER_bimodal_dataset/Corpus-Audio-162", "/home/soms/EmotionMusic/MER_bimodal_dataset/Wavs")
	#extract_features("/home/soms/EmotionMusic/bimodal/Wavs", "/home/soms/EmotionMusic/bimodal/Features")
	create_k_fold("/home/satyaki/Emotion/bimodal/Wavs", "/home/satyaki/Emotion/bimodal/5_Fold_Merge", "/home/satyaki/Emotion/bimodal/annotations.csv", 5)
