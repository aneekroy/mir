import math
import os
import numpy as np
import tensorflow as tf
import Image

TRAIN_PERCENT = 0.8
VALID_PERCENT = 0.1

def _int64_feature(val):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [val]))

def _bytes_feature(val):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [val]))

def convert_to_tfrecord(recordsdir, tfdir):
	outfilenametrain = tfdir + "/train.tfrecords"
	#outfilenamevalid = tfdir + "/valid.tfrecords"
	#outfilenametest = tfdir + "/test.tfrecords"
	trainwriter = tf.python_io.TFRecordWriter(outfilenametrain)
	#validwriter = tf.python_io.TFRecordWriter(outfilenamevalid)
	#testwriter = tf.python_io.TFRecordWriter(outfilenametest)
	traincount = 0
	#validcount = 0
	#testcount = 0
	trainlogger = open("/home/soms/EmotionMusic/Model2_SongWise/log-train.txt", "w")
	#validlogger = open("/home/soms/EmotionMusic/Model2/log-valid.txt", "w")
	#testlogger = open("/home/soms/EmotionMusic/Model2/log-test.txt", "w")
	for _, _, files in os.walk(recordsdir):
		for file in files:
			print(file)
			image_path = recordsdir + "/" + file
			image = np.asarray(Image.open(image_path).convert('L'))
			rows = image.shape[0]
			cols = image.shape[1]
			depth = 1
			image_raw = image.tostring()
			fileindex = int(file[:file.find("_")])
			classindex = math.trunc((math.ceil(fileindex/30.))) - 1
			print ("File %s Label %d Rows %d Cols %d Depth %d"%(file, classindex, rows, cols, depth))
			sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(classindex)), "image_raw":_bytes_feature(image_raw)}))
			trainwriter.write(sample.SerializeToString())
					#print("Converted train image %s"%(file))
			trainlogger.write(file + ":-" + str(classindex) + "\n")
			traincount += 1
			'''
				for file in validset:
					print(file)
					image_path = recordsdir + "/" + label + "/" + file
					image = np.asarray(Image.open(image_path))
					rows = image.shape[0]
					cols = image.shape[1]
					depth = image.shape[2]
					image_raw = image.tostring()
					sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(index)), "image_raw":_bytes_feature(image_raw)}))
					validwriter.write(sample.SerializeToString())
					#print("Converted train image %s"%(file))
					validlogger.write(file + ":-" + label + "\n")
				
				for file in testset:
					print(file)
					image_path = recordsdir + "/" + label + "/" + file
					image = np.asarray(Image.open(image_path).convert('L'))
					rows = image.shape[0]
					cols = image.shape[1]
					depth = 1
					image_raw = image.tostring()
					sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(index)), "image_raw":_bytes_feature(image_raw)}))
					testwriter.write(sample.SerializeToString())
					#print("Converted test image %s"%(file))
					testlogger.write(file + ":-" + label + "\n")
			'''
	print("Train:" + str(traincount))
	#print("Valid:" + str(validcount))
	#print("Test:" + str(testcount))
if __name__ == "__main__":
	convert_to_tfrecord("/home/soms/EmotionMusic/Spec_Feat/Train", "/home/soms/EmotionMusic/Model2_SongWise/Data")