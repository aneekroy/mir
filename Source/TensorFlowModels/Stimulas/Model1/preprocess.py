import os
import math
import numpy as np
import ffmpy
from python_speech_features import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import Image

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


'''
Extracts mfcc and delta features from the wav files and 
'''
def extract_features(wavdir, featuredir):
	for _,_, files in os.walk(wavdir):
		for file in files:
			fileindex = int(file[:file.find(".wav")])
			print fileindex, math.trunc(math.ceil(fileindex/30.))
			classindex = math.trunc((math.ceil(fileindex/30.)))
			classdir = "Class" + str(classindex)
			if not os.path.exists(os.path.join(featuredir, "Class" + str(classindex))):
				os.makedirs(os.path.join(featuredir,classdir))
				print("New folder created %s"%(os.path.join(featuredir, classdir)))
			print("File %s"%file)
			filename = file[:file.find(".wav")]
			(rate, sig) = wav.read(os.path.join(wavdir, file))
			sig = sig[:, 0]
			log_energy_feat = normalize(logfbank(sig, rate, nfilt = 32, nfft = 1024))
			#delta_feat = normalize(np.asarray(delta(log_energy_feat, 2)))
			#delta_delta_feat = normalize(np.asarray(delta(delta_feat, 2)))
			#image = np.dstack([log_energy_feat, delta_feat, delta_delta_feat])
			print log_energy_feat.shape
			image = log_energy_feat
			max, min = image.max(), image.min()
			for i in range(len(image)):
				for j in range(len(image[i])):
						image[i][j] = (image[i][j] - min)/(max - min) * 255
			bound = int(image.shape[0]/32)
			image = np.split(image[: 32 * bound, : ] , bound , axis = 0)
			index = 1
			for image_segment in image:	
				plt.clf()
				plt.imsave(os.path.join(featuredir + "/" + classdir, filename + "_" + str(index) + ".jpg"), image_segment.T, cmap = "spectral")
				index += 1


if __name__ == "__main__":
	mp3dir = "/home/soms/EmotionMusic/emotion/emotion-train"
	wavdir = "/home/soms/EmotionMusic/WavFiles"
	specdir = "/home/soms/EmotionMusic/Spectograms"
	featuredir = "/home/soms/EmotionMusic/New_Spectograms"
	#convert_mp3_to_wav(mp3dir, wavdir)
	#convert_wav_to_spec(wavdir, specdir)
	extract_features(wavdir, featuredir)