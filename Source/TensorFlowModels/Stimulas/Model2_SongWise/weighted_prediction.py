import os
import sys

CLASS_LABELS = {"H":0, "S":1, "T":2, "F/A":3}

def get_accuracy(infilehandle, outfilehandle, run_length_clip_skip = 1, run_length_tolerance = None):
	f = open(infilehandle, "r")
	lines = f.readlines()
	accuracy = 0
	run_length = {}
	tot_length = {}
	for line in lines:
		print("................................")
		line = line.split()
		label = int(line[-1])
		example_index = line[0]
		print(example_index)
		line = line[1:-1]
		print("Length of clip: " + str(len(line)))
		for lab in CLASS_LABELS:
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

	candidates = []
	M = 0
	for lab in CLASS_LABELS:
		if run_length[lab] > M:
			M = run_length[lab]
			candidates = [lab]
		elif run_length[lab] == M:
			candidates.append(lab)
	if run_length_tolerance != None:
		for lab in CLASS_LABELS:
			if run_length[lab] >= run_length_tolerance * len(line):
				if lab not in candidates:
					candidates.append(lab)
	print("Candidates ", candidates)