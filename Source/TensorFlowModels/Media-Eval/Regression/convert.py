import math
import os
import numpy as np
import tensorflow as tf
import Image

TRAIN_PERCENT = 0.8
VALID_PERCENT = 0.1

def _int64_feature(val):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [val]))

def _float_feature(val):
	return tf.train.Feature(float_list = tf.train.FloatList(value = [val]))

def _bytes_feature(val):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [val]))

def convert_to_tfrecord_train(recordsdir, tfdir):
	outfilenametrain = tfdir + "/train.tfrecords"
	trainwriter = tf.python_io.TFRecordWriter(outfilenametrain)
	traincount = 0
	labelfile = open("/home/soms/EmotionMusic/MediaEval/new-label-file.txt", "r")
	arousal = {}
	for line in labelfile.readlines():
		line = line.split()
		arousal[line[0][:line[0].find("_")]] = float(line[1])
	for _, _, files in os.walk(recordsdir):
		for file in files:
			image_path = recordsdir + "/" + file
			image = np.asarray(Image.open(image_path).convert('L'))
			rows = image.shape[0]
			cols = image.shape[1]
			depth = 1
			image_raw = image.tostring()
			classindex = arousal[file[:file.find(".wav")]]
			print ("File %s Label %0.2f Rows %d Cols %d Depth %d"%(file, classindex, rows, cols, depth))
			sample = tf.train.Example(features = tf.train.Features(feature = {"label":_float_feature(classindex), "image_raw":_bytes_feature(image_raw)}))
			trainwriter.write(sample.SerializeToString())
					#print("Converted train image %s"%(file))
			#trainlogger.write(file + ":-" + str(classindex) + "\n")
			traincount += 1
	print("Train:" + str(traincount))
'''
def convert_to_tfrecord_test(recordsdir, tfdir):
	testcount = 0
	for _, _, files in os.walk(recordsdir):
		for file in files:
			outfilenametest = tfdir + "/" + str(file[:file.find(".png")]) + ".tfrecords"
			testwriter = tf.python_io.TFRecordWriter(outfilenametest)
			image_path = recordsdir + "/" + file
			image = np.asarray(Image.open(image_path).convert('L'))
			classindex = int(file[ : file.find("_")])
			image = np.array_split(image, math.ceil(image.shape[0]/60.))
			index = 1
			for image_segment in image:	
				image_segment = image_segment[image_segment.shape[0] - 57 : ][:56]
				image_segment = image_segment.T
				rows = image_segment.shape[0]
				cols = image_segment.shape[1]
				depth = 1
				image_raw = image_segment.tostring()
				print ("File %s Segment %d Label %d Rows %d Cols %d Depth %d"%(file, index, classindex, rows, cols, depth))
				sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(classindex)), "image_raw":_bytes_feature(image_raw)}))
				testwriter.write(sample.SerializeToString())
				#print("Converted train image %s"%(file))
				index += 1
			testcount += 1

	print("Test:" + str(testcount))

'''
def convert_to_tfrecord_test(recordsdir, tfdir):
	outfilenametest = tfdir + "/test.tfrecords"
	testwriter = tf.python_io.TFRecordWriter(outfilenametest)
	testcount = 0
	labelfile = open("/home/soms/EmotionMusic/MediaEval/new-label-file.txt", "r")
	arousal = {}
	for line in labelfile.readlines():
		line = line.split()
		arousal[line[0][:line[0].find("_")]] = float(line[1])
	for _, _, files in os.walk(recordsdir):
		for file in files:
			image_path = recordsdir + "/" + file
			image = np.asarray(Image.open(image_path).convert('L'))
			rows = image.shape[0]
			cols = image.shape[1]
			depth = 1
			image_raw = image.tostring()
			classindex = float(arousal[file[:file.find(".wav")]])
			print ("File %s Label %f Rows %d Cols %d Depth %d"%(file, classindex, rows, cols, depth))
			sample = tf.train.Example(features = tf.train.Features(feature = {"label":_float_feature(classindex), "image_raw":_bytes_feature(image_raw)}))
			testwriter.write(sample.SerializeToString())
			testcount += 1
	print("Test:" + str(testcount))


if __name__ == "__main__":
	#convert_to_tfrecord_train("/home/soms/EmotionMusic/MediaEval/Spec_Feat/Train", "/home/soms/EmotionMusic/MediaEval/Data")
	convert_to_tfrecord_test("/home/soms/EmotionMusic/MediaEval/Spec_Feat/Test", "/home/soms/EmotionMusic/MediaEval/Data")