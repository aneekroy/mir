import matplotlib.pyplot as plt
import os
import Image
import scipy.io.wavfile as wav
import numpy as np
from numpy.lib import stride_tricks
from python_speech_features import *

def normalize(A):
	M = [[x]*A.shape[1] for x in np.mean(A, axis = 1)]
	S = [[x]*A.shape[1] for x in np.std(A, axis = 1) if x is not 0]
	A = (A - M)
	for i in range(len(A)):
		for j in range(len(A[i])):
			if S[i][j] != 0.0:
				A[i][j] = A[i][j]/S[i][j]
	return A

def sfft(signal, framesize, overlap = 0.5, window = np.hanning):
	win = window(framesize)
	hopsize = int(framesize - np.floor(overlap * framesize))
	samples = np.append(np.zeros(np.floor(framesize/2.0)), signal)
	cols = np.ceil((len(samples) - framesize) / float(hopsize)) + 1
	samples = np.append(samples, np.zeros(framesize))
	frames = stride_tricks.as_strided(samples, shape = (cols, framesize),strides = (samples.strides[0]*hopsize, samples.strides[0])).copy()
	frames *= win
	return np.fft.rfft(frames)

def logscale_spec(spec, sr = 44100, factor = 20., alpha = 1.0, f0 = 0.9, fmax = 1):
	spec = spec[: , 0:256]
	timebins, freqbins = np.shape(spec)
	scale = np.linspace(0, 1 ,freqbins)
	scale = np.array(map(lambda x:  x * alpha if x <= f0 else (fmax - alpha * f0)/(fmax - f0) * (x - f0) + alpha * f0, scale))
	scale *= (freqbins - 1)/max(scale)
	newspec = np.complex128(np.zeros([timebins, freqbins]))
	allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins + 1])
	freqs = [0.0 for i in range(freqbins)]
	totw = [0.0 for i in range(freqbins)]
	for i in range(0, freqbins):
		if(i < 1 or i + 1 >= freqbins):
			newspec[:, i] += spec[:, i]
			freqs[i] += allfreqs[i]
			totw[i] += 1.0
			continue
		else:
			w_up = scale[i] - np.floor(scale[i])
			w_down = 1 - w_up
			j = int(np.floor(scale[i]))
			newspec[:,j] += w_down * spec[:, i]
			freqs[j] += w_down * allfreqs[i]
			totw[j] += w_down
			newspec[:, j + 1] += w_up * spec[:, i]
			freqs[j + 1] += w_up * allfreqs[i]
			totw[j + 1] += w_up
	for i in range(len(freqs)):
		if(totw[i] >= 1e-6):
			freqs[i] /= totw[i]
	return newspec, freqs

def plotsfft(audiopath, binsize = 2**10, plotpath = None, colormap = "jet", channel = 0, name = 'tmp.jpg', alpha = 1, offset = 0):
	samplerate, samples = wav.read(audiopath)
	samples = samples[:,channel]
	s = sfft(samples, binsize)
	sshow, freq = logscale_spec(s, factor = 1, sr = samplerate, alpha = alpha)
	sshow = sshow[2:, :]
	ims = 20.*np.log10(np.abs(sshow)/10e-6)
	timebins, freqbins = np.shape(ims)
	ims = np.transpose(ims)
	ims = ims[0:256, offset : offset + 768]
	plt.imshow(ims, cmap = "spectral")
	plt.colorbar()
	plt.show()
if __name__ == "__main__":
	audiopath = "/home/soms/EmotionMusic/WavFiles/032.wav"
	plotsfft(audiopath)
	'''
	rate, signal = wav.read(audiopath)
	print signal[:,0].shape[0]/rate
	plt.clf()
	mfcc = normalize(mfcc(signal, rate, nfft = 1024))
	melspec = logfbank(signal, rate, nfilt = 32, nfft = 1024)
	delta_feat = np.asarray(delta(melspec, 2))
	delta_delta_feat = np.asarray(delta(delta_feat, 2))
	image = normalize(melspec)
	
	max, min = image.max(), image.min()
	for i in range(len(image)):
		for j in range(len(image[i])):
			image[i][j] = (image[i][j] - min)/(max - min) * 255
	
	plt.imshow(image.T, origin = "lower", aspect = "auto", cmap = "gray")
	#plt.imsave('tmp.jpg', melspec)
	plt.colorbar()
	plt.show()
	#plt.imshow(im, origin = "lower", aspect = "auto", cmap = "jet")
	'''