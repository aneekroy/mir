import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import Image
import sklearn
import random
import csv

def normalize(A):
    #print np.mean(A), np.std(A)
    if np.std(A) != 0:
        A = (A - np.mean(A)) / np.std(A)
    return A

def write_tempo_record(file, wavdir, outdir):
    pass


def write_spec_record(file, wavdir, outdir):
    filename = file[:file.find(".wav")]
    print("preparing %s"%filename)
    y, sr = librosa.load(os.path.join(wavdir, file))
    y = np.array_split(y, 6)
    index = 1
    for seg in y:
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels = 128, fmax = 8000)
        log_S = librosa.logamplitude(S, ref_power=np.max)
        M = np.max(np.abs(log_S))
        for i in range(len(log_S)):
            for j in range(len(log_S[i])):
                log_S[i,j] = float(log_S[i,j])/M
        log_S = normalize(log_S)
        M = np.max(np.abs(log_S))
        m = np.min(np.abs(log_S))
        normedS = (((log_S - log_S.min()) / (log_S.max() - log_S.min())) * 255.9).astype(np.uint8)
        print normedS.shape
        img = Image.fromarray(normedS)
        img.save(os.path.join(outdir, filename + "_" + str(index) + ".png"))
        index += 1
    #img = np.asarray(Image.open(os.path.join(featuredir, "A002_1.png")).convert("L"))
    #print img, img.shape

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
						write_spec_record(file, wavdir, featuredir + "/Fold" + str(index) + "/Train")
				else:
					for file in stratified_samples[label][i]:
						write_spec_record(file, wavdir, featuredir + "/Fold" + str(index) + "/Test")

if __name__ == "__main__":
    create_k_fold("/home/satyaki/Emotion/bimodal/Wavs", "/home/satyaki/Emotion/bimodal/5_Fold_En", "/home/satyaki/Emotion/bimodal/annotations.csv", 5)
