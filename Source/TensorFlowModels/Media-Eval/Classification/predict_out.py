import os
import sys

CLASS_LABELS = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7}

def get_accuracy(file, run_length_enable = True, run_length_clip_skip = 1, run_length_tolerance = None, tot_clips_tolerance = None):
	f = open(file, "r")
	lines = f.readlines()
	accuracy = 0
	min = 999
	max = 0
	run_length = {}
	tot_length = {}
	for line in lines:
		print("................................")
		line = line.split()
		label = int(line[-1])
		example_index = line[0]
		print(example_index)
		line = line[1:-1]
		if len(line) < min:
			min = len(line)
		if len(line) > max:
			max = len(line)
		print("Length of clip: " + str(len(line)))
		for lab in CLASS_LABELS:
			#print("Label " + lab)
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
			tot_length[lab] = 0
			while index < len(line):
				if line[index] == lab:
					tot_length[lab] += 1
				index += 1
			if run_length_enable:
				print("Label: %s MRL: %d TCC: %d"%(lab, run_length[lab], tot_length[lab]))
			else:
				print("Label: %s TCC: %d"%(lab, tot_length[lab]))

		#for lab in CLASS_LABELS:
		#	print(lab + ":" + str(run_length[lab]))
		#print("Actual:" + str(label))
		if run_length_enable:
			candidates = []
			M = 0
			for lab in CLASS_LABELS:
				if run_length[lab] > M:
					M = run_length[lab]
			for lab in CLASS_LABELS:
				if run_length[lab] == M:		
					candidates.append(lab)
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
		for lab in CLASS_LABELS:
			if tot_length[lab] == L:
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
	print("Accuracy = %.3f"%accuracy)
	print("Min:", min)
	print("Max:", max)

if __name__ == "__main__":
	get_accuracy("/home/soms/EmotionMusic/MediaEval_Classification/Outputs/output_sequence_2.txt", run_length_tolerance = None, tot_clips_tolerance = 0.25)
