import os
import ffmpy
from python_speech_features import *
import scipy.io.wavfile as wav
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import random
import Image

def normalize(A):
    #print np.mean(A), np.std(A)
    if np.std(A) != 0:
        A = (A - np.mean(A)) / np.std(A)
    return A

TRAIN_PERCENT = 0.8



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

M = 99999

def write_record(file, outdir, label, labelfile, i):
	global M    
	filename = file[file.rfind("/") + 1:file.find(".wav")]
	print("preparing %s"%filename)
	y, sr = librosa.load(file)
	#print y.shape
	y = np.array_split(y, 6)
	for seg in y:	
		S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels = 128, fmax = 8000)	
		print S.shape	
		if i == 0:	
			if S.shape[1] < M:
				M = S.shape[1]
		S = S.T
		
		index = 1
		log_S = librosa.logamplitude(seg, ref_power=np.max)
        	M = np.max(np.abs(log_S))
        	for k in range(len(log_S)):
            		for j in range(len(log_S[k])):
                		log_S[k,j] = float(log_S[k,j])/M
       		log_S = normalize(log_S)
        	M = np.max(np.abs(log_S))
        	m = np.min(np.abs(log_S))
       		normedS = (((log_S - log_S.min()) / (log_S.max() - log_S.min())) * 255.9).astype(np.uint8)
        	print normedS.shape
        	img = Image.fromarray(normedS)
        	#img.save(os.path.join(outdir, filename + "_" + str(index) + ".png"))
		
		if i == 0:
			labelfile.write(filename + "_" + str(index) + " " + label + "\n")
			print(filename + "_" + str(index) + " " + label)
        	index += 1
	

def create_k_folds(featuredir, sampledict, k):
	labelfile = open("/home/satyaki/Emotion/Soundtrack/labels.txt", "w")
	stratified_samples = dict()	
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
						write_record(file, featuredir + "/Fold" + str(index) + "/Train", label, labelfile, index)
				else:
					for file in stratified_samples[label][i]:
						write_record(file, featuredir + "/Fold" + str(index) + "/Test", label, labelfile, index)
		

def prep_data(dirpath):
	random.seed(7)
	sampledict = dict()    
	for _,classes,_ in os.walk(dirpath):
		for Class in classes:
			for _,_,files in os.walk(os.path.join(dirpath, Class)):
				for file in files:
					filepath = os.path.join(dirpath, Class) + "/" + file            		
					if Class not in sampledict:
						sampledict[Class] = [filepath]
					else:
						sampledict[Class].append(filepath)
	return sampledict


if __name__ == "__main__":
	sampledict = prep_data("/home/satyaki/Emotion/Soundtrack/WavFiles")
	create_k_folds("/home/satyaki/Emotion/Soundtrack/10_Fold", sampledict, 5)
	print("min", M)

