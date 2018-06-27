import matplotlib.pyplot as plt
import numpy as np
import Image
import tensorflow as tf

def normalize(A):
	M = [[x]*A.shape[1] for x in np.mean(A, axis = 1)]
	S = [[x]*A.shape[1] for x in np.std(A, axis = 1) if x is not 0]
	A = (A - M)
	for i in range(len(A)):
		for j in range(len(A[i])):
			if S[i][j] != 0.0:
				A[i][j] = A[i][j]/S[i][j]
	return A

def example():
	mat = np.genfromtxt("/home/soms/EmotionMusic/Hindi_Raga_Spec/23.Kirwani/03 - Raga Kirwani Slow And Fast Gats_inst1.txt", delimiter = ",")

	plt_x = np.arange(mat.shape[1])
	plt_y = np.arange(mat.shape[0])
	plt_z = mat
	z_min = plt_z.min()
	z_max = plt_z.max()
	#plt_z = plt_z.T

	color_map = plt.cm.jet
	plt.clf()
	plt.pcolor(plt_x, plt_y, plt_z, cmap = color_map, vmin = 0, vmax = 1)
	plt.axis([plt_x.min(), plt_x.max(), plt_y.min(), plt_y.max()])
	ax = plt.gca()
	ax.set_aspect("equal")
	figure = plt.gcf()
	plt.show()

def load_image():
	file = '/home/soms/EmotionMusic/Hindi_Raga_Spec/1.Kanada/01. Raga Asavari Todi_vocal1.txt'
	image = normalize(np.genfromtxt(file, delimiter = ","))
	max, min = image.max(), image.min()
	for i in range(len(image)):
		for j in range(len(image[i])):
			image[i][j] = (image[i][j] - min)/(max - min) * 255
	print image.shape, image.min(), image.max()
	plt.imshow(image, cmap = "gray")
	plt.show()
	'''
	filename_queue = tf.train.string_input_producer([file])
	reader = tf.WholeFileReader()
	key, val = reader.read(filename_queue)
	img = tf.image.decode_jpeg(val)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
  		coord = tf.train.Coordinator()
	  	threads = tf.train.start_queue_runners(coord=coord)

		for i in range(1):
			image = img.eval()
			print image.shape
			Image.fromarray(np.asarray(image)).show()
	coord.request_stop()
  	coord.join(threads)
  	sess.close()
  	'''

if __name__ == "__main__":
	load_image()