import os
import numpy as np
import tensorflow as tf
import Image

TRAIN_PERCENT = 0.8

def _int64_feature(val):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [val]))

def _bytes_feature(val):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [val]))

def convert_to_tfrecord(recordsdir, tfdir):
	outfilenametrain = dir + "/train.tfrecords"
	outfilenametest = dir + "/test.tfrecords"
	trainwriter = tf.python_io.TFRecordWriter(outfilenametrain)
	testwriter = tf.python_io.TFRecordWriter(outfilenametest)
	traincount = 0
	testcount = 0
	trainlogger = open("/home/soms/EmotionMusic/Model1/log-train.txt", "w")
	testlogger = open("/home/soms/EmotionMusic/Model1/log-test.txt", "w")
	with _,classes, _ in os.walk(recordsdir):
		for label in classes:
			for _, _, files in os.walk(os.path.join(recordsdir, label)):
				size = len(files)
				trainset = files[: TRAIN_PERCENT * size]
				testset = files[TRAIN_PERCENT * size :]
				traincount += len(trainset)
				testcount += len(testset) 
				for file in trainset:
					image_path = recordsdir + "/" + label + "/" + file
					image = np.asarray(Image.open(image_path))
					rows = image.shape[0]
					cols = image.shape[1]
					depth = image.shape[2]
					image_raw = image.tostring()
					sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(label)), "image_raw":_bytes_feature(image_raw)}))
					trainwriter.write(sample.SerializeToString())
					print("Converted train image %d"%(i+1))
					trainlogger.write("Converted train image %s\n"%(file))
				for file in testset:
					image_path = recordsdir + "/" + label + "/" + file
					image = np.asarray(Image.open(image_path))
					rows = image.shape[0]
					cols = image.shape[1]
					depth = image.shape[2]
					image_raw = image.tostring()
					sample = tf.train.Example(features = tf.train.Features(feature = {"height":_int64_feature(rows), "width":_int64_feature(cols), "depth":_int64_feature(depth), "label":_int64_feature(int(label)), "image_raw":_bytes_feature(image_raw)}))
					testwriter.write(sample.SerializeToString())
					print("Converted test image %d"%(i+1))
					testlogger.write("Converted test image %s\n"%(file))
if __name__ == "__main__":
	convert_to_tfrecord("/home/soms/EmotionMusic/Features", "/home/soms/EmotionMusic/Model1/Data")