import os
import numpy as np

f = open("/home/soms/EmotionMusic/MediaEval/outputs/predictions.txt", "r")
predicted = {}
actual = {}
for line in f.readlines():
	line = line.split()
	filename = line[0][:line[0].find(".")]
	#print filename	
	if filename not in predicted:
		predicted[filename] = [float(line[2])]
		actual[filename] = float(line[1])
	else:
		predicted[filename].append(float(line[2]))
	
y_m = 0.0
for p in predicted:
	predicted[p] = np.mean(np.asarray(predicted[p], dtype = np.float32))
	y_m += predicted[p]
y_m = float(y_m) / len(predicted.keys())

r_m= 0.0
r_y = 0.0
for p in predicted:
	r_y += np.square(predicted[p] - actual[p])
	r_m += np.square(actual[p] - y_m)

rmse = np.sqrt(float(r_y) / len(predicted.keys()))
r_squared = 1 - float(r_y) / r_m

print("RMSE:%.4f R-Squared:%.4f"%(rmse, r_squared))

