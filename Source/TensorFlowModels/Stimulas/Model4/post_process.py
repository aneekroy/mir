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
	conf_mat = np.zeros((4,4), dtype = np.float32)
	for line in lines:
		line = line.split()
		actual = int(line[-1])
		counts = {}
		for l in line[1:-1]:
			conf_mat[actual, int(CLASS_LABELS[l])] += 1
	s = np.sum(conf_mat)
	#print s
	for i in range(len(conf_mat)):
		for j in range(len(conf_mat[i])):
			conf_mat[i,j] /= s
			conf_mat[i,j] *= 100.
			print "%.2f"%conf_mat[i,j],
		print("\n")
	#print np.sum(conf_mat)
	for i in range(len(conf_mat)):
		act = conf_mat[i,i]
		s1 = 0
		s2 = 0
		for j in range(len(conf_mat[i])):
				s1 += conf_mat[j,i]
				s2 += conf_mat[i,j]
		pr = float(act)/s1
		re = float(act)/s2
		print("%d Precision:%.2f Recall:%.2f F-Measure:%.2f"%(i, pr, re, float(2 * pr * re)/(pr + re)))


def get_accuracy(file, run_length_enable = True, run_length_clip_skip = 1, threshold = 0.5):
	f = open(file, "r")
	lines = f.readlines()
	segment_accuracy = get_segment_accuracy(file)
	accuracy = 0
	compute_confusion(file)
	conf_mat = np.zeros((4, 4), dtype = np.float32)
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
		line = line[1:-1]
		if len(line) < min:
			min = len(line)
		if len(line) > max:
			max = len(line)
		print("Length of clip: " + str(len(line)))
		for lab in sorted(CLASS_LABELS):
			#print("Label " + lab)
			# Compute the maximum run length for each label
			if run_length_enable == True:
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
			if run_length_enable:
				print("Label: %s MRL: %d TCC: %d PCC: %.2f"%(lab, run_length[lab], tot_length[lab], float(tot_length[lab]) / len(line)))
			else:
				print("Label: %s TCC: %d PCC: %.2f"%(lab, tot_length[lab], float(tot_length[lab]) / len(line)))
				
		candidates = []
		MaxCount = 0
		for lab in CLASS_LABELS:
			if tot_length[lab] >= MaxCount:
				MaxCount = tot_length[lab]
		for lab in CLASS_LABELS:
			if tot_length[lab] == MaxCount:
				candidates.append(lab)
		# If more than 1 label has the same maximum run length, use majority voting.
		
		if len(candidates) >= 1:
			MV = 0
			new_candidates = []
			for lab in candidates:
				if run_length[lab] >= MV:
					MV = run_length[lab]
			for lab in candidates:
				if run_length[lab] == MV:
					new_candidates.append(lab)
		
		for index in range(len(new_candidates)):
			new_candidates[index] = CLASS_LABELS[new_candidates[index]]
		print("Prediction:%s Actual:%d"%(new_candidates, label))
		if len(new_candidates) > 1:
			print("=*100\nCaught\n=*100")
		candidates = new_candidates[0]
		print("................................")
		if label == candidates:
			accuracy += 1
		conf_mat[label, candidates] += 1
	accuracy = float(accuracy) / len(lines)
	print("Segment Accuracy = %.3f"%segment_accuracy)
	print("Accuracy = %.3f"%accuracy)
	for i in range(len(conf_mat)):
		for j in range(len(conf_mat[i])):
			conf_mat[i,j] /= 74.
			conf_mat[i,j] *= 100 
			print "%.2f"%conf_mat[i,j],
		print("\n")
	
	for i in range(len(conf_mat)):
		act = conf_mat[i,i]
		s1 = 0
		s2 = 0
		for j in range(len(conf_mat[i])):
				s1 += conf_mat[j,i]
				s2 += conf_mat[i,j]
		pr = float(act)/s1
		re = float(act)/s2
		print("%d Precision:%.2f Recall:%.2f F-Measure:%.2f"%(i, pr, re, float(2 * pr * re)/(pr + re)))
	return accuracy
	
if __name__ == "__main__":
	print(get_accuracy("/home/soms/EmotionMusic/Model4/Outputs/sequence(63.5).txt"))