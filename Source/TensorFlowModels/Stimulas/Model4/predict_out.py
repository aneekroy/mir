import os
import sys
import numpy as np

CLASS_LABELS = {"H":0, "S":1, "T":2, "F/A":3}

def get_segment_accuracy(file):
	f = open(file, "r")
	lines = f.readlines()
	tot_cnt = 0
	correct_cnt = 0
	for line in lines:
		line = line.split()
		actual = line[-1]
		line = line[1:-1]
		tot_cnt += len(line)
		for l in line:
			if CLASS_LABELS[l] == int(actual):
				correct_cnt += 1
	segment_acc = float(correct_cnt) / tot_cnt
	return segment_acc
	
	
def compute_confusion(file):
	f = open(file, "r")
	lines = f.readlines()
	conf_mat = np.zeros((4,4), dtype = np.int32)
	for line in lines:
		line = line.split()
		actual = int(line[-1])
		counts = {}
		for l in line[:-1]:
			if CLASS_LABELS[l] not in counts:
				counts[CLASS_LABELS[l]] = 1
			else:
				counts[CLASS_LABELS[l]] += 1
		M = 0
		predicted = 0
		for c in counts:
			if counts[c] > M:
				M = counts[c]
				predicted = c
		conf_mat[predicted, actual] += 1
	
	for i in range(len(conf_mat)):
		for j in range(len(conf_mat[i])):
			print conf_mat[i,j],
		print("\n")
	
	for i in range(len(conf_mat)):
		act = conf_mat[i,i]
		s1 = 0
		s2 = 0
		for j in range(len(conf_mat[i])):
				s1 += conf_mat[i,j]
				s2 += conf_mat[j,i]
		pr = float(act)/s1
		re = float(act)/s2
		print("%d Precision:%.2f Recall:%.2f F-Measure:%.2f"%(i, pr, re, float(2 * pr * re)/(pr + re)))


def get_accuracy(file, run_length_enable = True, run_length_clip_skip = 1, run_length_tolerance = None, tot_clips_tolerance = None):
	f = open(file, "r")
	lines = f.readlines()
	segment_accuracy = get_segment_accuracy(file)
	accuracy = 0
	compute_confusion(file)
	min = 999
	max = 0
	run_length = {}
	tot_length = {}
	for line in lines:
		print("................................")
		line = line.split()
		label = int(line[-1])
		#example_index = line[0]
		#print(example_index)
		line = line[:-1]
		if len(line) < min:
			min = len(line)
		if len(line) > max:
			max = len(line)
		print("Length of clip: " + str(len(line)))
		for lab in CLASS_LABELS:
			#print("Label " + lab)
			# Compute the maximum run length for each label
			if run_length_enable == False or run_length_enable == True:
				run_length[lab] = 0
				index = 0
				while index < len(line):
					# Find the first occurence of a run-length of the label
					while index < len(line):
						if line[index] == lab:
							break
						index += 1
					first = index
					last = index
					#print("First:" + str(first))
					if index == len(line):
						break
					while index < len(line):
						if line[index] == lab:
							last = index
							index += 1
						else:
							start = index
							end = index + run_length_clip_skip
							if start >= len(line):
								break
							if end >= len(line):
								end = len(line) - 1
							while start <= end:
								if line[start] == lab:
									last = start
								 	index = start
									break
								start += 1
							if start == end + 1:
								index = start
								break

					#print("Last:" + str(last))
					cnt = last - first + 1
					#print("Count:" + str(cnt))
					if cnt > run_length[lab]:
						run_length[lab] = cnt
			index = 0
			# Compute the total clip count for each label
			tot_length[lab] = 0
			while index < len(line):
				if line[index] == lab:
					tot_length[lab] += 1
				index += 1
			#if run_length_enable:
			print("Label: %s MRL: %d TCC: %d PCC: %.2f"%(lab, run_length[lab], tot_length[lab], float(tot_length[lab]) / len(line)))
			#else:
			#	print("Label: %s TCC: %d PCC: %.2f"%(lab, tot_length[lab], float(tot_length[lab]) / len(line)))

		#for lab in CLASS_LABELS:
		#	print(lab + ":" + str(run_length[lab]))
		#print("Actual:" + str(label))
		if run_length_enable:
			candidates = []
			#M = 0
			#for lab in CLASS_LABELS:
			#	if run_length[lab] > M:
			#		M = run_length[lab]
			#for lab in CLASS_LABELS:
			#	if run_length[lab] == M:		
			#		candidates.append(lab)
			if run_length_tolerance != None:
				for lab in CLASS_LABELS:
					tolernace_rl = (run_length_tolerance * len(line))
					if tolernace_rl < 5:
						tolernace_rl = 5
					if run_length[lab] >= tolernace_rl:
						if lab not in candidates:
							candidates.append(lab)
			print("Candidates ", candidates)
		else:
			candidates = []
		L = 0
		prediction = []
		for lab in CLASS_LABELS:
			if tot_length[lab] > L:
				L = tot_length[lab]
		cnt = 0
		for lab in CLASS_LABELS:
			if tot_length[lab] > 0:
				cnt += 1
		print("#Classes predicted:%d"%cnt)
		if L >= float(cnt) / 4 * len(line):
			for lab in CLASS_LABELS:
				if tot_length[lab] == L:
					prediction.append(CLASS_LABELS[lab])
		else:
			M = 0
			for lab in CLASS_LABELS:
				if run_length[lab] > M:
					M = run_length[lab]
			for lab in CLASS_LABELS:
				if run_length[lab] == M:
					prediction.append(CLASS_LABELS[lab])
		if tot_clips_tolerance != None:
			for lab in CLASS_LABELS:
				if tot_length[lab] >= (tot_clips_tolerance * len(line)):
					if CLASS_LABELS[lab] not in prediction:
						prediction.append(CLASS_LABELS[lab])
		for candidate in candidates:
			prediction.append(CLASS_LABELS[candidate])
		prediction = set(prediction)
		print("Prediction:%s Actual:%d"%(prediction, label))
		print("................................")
		if label in prediction:
			accuracy += 1
	accuracy = float(accuracy)/len(lines)
	print("Segment Accuracy = %.3f"%segment_accuracy)
	print("Accuracy = %.3f"%accuracy)
	print("Min:", min)
	print("Max:", max)

if __name__ == "__main__":
	get_accuracy("/home/soms/EmotionMusic/Model4/Outputs/sequence(63.5).txt", run_length_enable = True, run_length_tolerance = None, tot_clips_tolerance = None)