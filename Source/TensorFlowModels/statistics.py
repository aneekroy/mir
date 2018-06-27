import numpy as np
import Image

labelfile = "/home/soms/EmotionMusic/image-label-file.txt"
statfile = "/home/soms/stat-file.txt"

f = open(labelfile,"r")
g = open(statfile, "w")
c = 0
for line in f.readlines():
	filename = line.split(":-")[0]
	img = np.asarray(Image.open(filename))
	'''
	buckets = {}
	for i in range(5):
		buckets[i] = 0
	#print len(img),len(img[0]) 
	for i in range(len(img)):
		for j in range(len(img[i])):
			if img[i][j] >= 1:
				buckets[4] += 1
			else:
				buckets[int(float(img[i][j])/0.2)] += 1
	#print("File %d: %d %d %d %d %d %d\n"%(c, buckets[0], buckets[1], buckets[2], buckets[3], buckets[4], sum(buckets.values())))
	try:
		g.write("File " + str(c) + ":" + str(buckets[0]) + " " + str(buckets[1]) + " " + str(buckets[2]) + " " + str(buckets[3]) + " " + str(buckets[4]) + "\n")
		print("Wrote record")
	except Exception:
		print("Failed to write record due to exception:%s"%Exception)
	'''
	c += 1
	print("File %d Max %d"%(c,img.max()))
