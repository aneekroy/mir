import math
import os
import numpy as np
import tensorflow as tf
import Image

TRAIN_PERCENT = 0.8
VALID_PERCENT = 0.1

labelfile = open("/home/soms/EmotionMusic/ModelRaga/new-label-file.txt", "r")
imageinputs = {}
for line in labelfile.readlines():
  	line = line.split(":-")
  	imageinputs[line[0]] = int(line[1])


def _int64_feature(val):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [val]))

def _bytes_feature(val):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [val]))

def convert_to_tfrecord(type, imagedict, recordsdir, tfdir):
	if type == "train":	
		outfilenametrain = tfdir + "/train.tfrecords"
		trainwriter = tf.python_io.TFRecordWriter(outfilenametrain)
		traincount = 0
		#trainlogger = open("/home/soms/EmotionMusic/Model2_SongWise/log-train.txt", "w")
		for _, _, files in os.walk(recordsdir):
			for file in files:
				image_path = recordsdir + "/" + file
				image = np.asarray(Image.open(image_path).convert('L'))
				rows = image.shape[0]
				cols = image.shape[1]
				depth = 1
				image_raw = image.tostring()
				classindex = imagedict[file[ : file.find(".jpg")]]
				print ("File %s Label %d Rows %d Cols %d Depth %d"%(file, classindex, rows, cols, depth))
				sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(classindex)), "image_raw":_bytes_feature(image_raw)}))
				trainwriter.write(sample.SerializeToString())
						#print("Converted train image %s"%(file))
				#trainlogger.write(file + ":-" + str(classindex) + "\n")
				traincount += 1
		print("Train:" + str(traincount))
	
	elif type =="test":
		outfilenametest = tfdir + "/test.tfrecords"
		testwriter = tf.python_io.TFRecordWriter(outfilenametest)
		testcount = 0
		#trainlogger = open("/home/soms/EmotionMusic/Model2_SongWise/log-train.txt", "w")
		for _, _, files in os.walk(recordsdir):
			for file in files:
				image_path = recordsdir + "/" + file
				image = np.asarray(Image.open(image_path).convert('L'))
				rows = image.shape[0]
				cols = image.shape[1]
				depth = 1
				image_raw = image.tostring()
				classindex = imagedict[file[ : file.find(".jpg")]]
				print ("File %s Label %d Rows %d Cols %d Depth %d"%(file, classindex, rows, cols, depth))
				sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(classindex)), "image_raw":_bytes_feature(image_raw)}))
				testwriter.write(sample.SerializeToString())
						#print("Converted train image %s"%(file))
				#trainlogger.write(file + ":-" + str(classindex) + "\n")
				testcount += 1
		print("Test:" + str(testcount))

if __name__ == "__main__":
	convert_to_tfrecord("test", imageinputs, "/home/soms/EmotionMusic/ModelRaga/Spectograms/Test", "/home/soms/EmotionMusic/ModelRaga/Data")
