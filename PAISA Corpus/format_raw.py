#coding: utf-8

import os
import argparse
import pandas as pd
import re
import numpy as np
from dfply import *
#from dplython import (DplyFrame, X, diamonds, select, sift, sample_n, nrow,
#    sample_frac, head, arrange, mutate, group_by, summarize, DelayFunction, left_join) 

parser = argparse.ArgumentParser(description='Enter folder name that contains subfolders for conll and kwic data. The names \
	in these subfolders should match to ensure that target words get paired to the correct sentences.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder with subfolders for conll and kwic data')

conll_names = ['TokenNum', "Word", "Lemma", "CPOS", "POS", "Features", "Head", "Dependency", "X1", "X2"]
dtypes = {'TokenNum':np.int32, "Word":np.dtype(str), "Lemma":np.dtype(str), "CPOS":np.dtype(str), "POS":np.dtype(str), "Features":np.dtype(str), "Head":np.dtype(int), "Dependency":np.dtype(str), "X1":np.dtype(str), "X2":np.dtype(str)}

foldername = parser.parse_args().data_folder[0]

# custom functions for dfply
#def _concat(X):
#	return [	

@make_symbolic
def concat(series):
	return ' '.join(str(x) for x in series)


# the sentence dataframe is where we will collect our processed data
sentences = pd.DataFrame()

if os.path.isdir(foldername):
	# we should have matching files in a conll and a kwic directory, make sure they are both there
	if "conll" in os.listdir(foldername) and "kwic" in os.listdir(foldername):
		conll_files = os.listdir(foldername + "/data/conll")
		kwic_files = os.listdir(foldername + "/data/kwic")
		match_files = list(set(conll_files) & set(kwic_files))
		if len(match_files) > 0:
			for f in match_files:
				filename = foldername + "/kwic/" + f
				with open(filename, encoding="utf-8") as in_file:
					sentences = in_file.readlines()
					# get sentence w/o target and target
					targets = []
					target_names = []
					for s in sentences:
						matches = re.search(r'(.*)?<(.*?)>(.*)', s)
						matches = matches.group(2).split()
						targets.append(matches)
					
					ntargets = len(targets[0])
					target_names = ["_target_" + str(t) for t in range(ntargets)]
					targets = pd.DataFrame(targets, columns=target_names)
					targets['SentenceNum'] = list(range(1,1+len(targets)))
				filename = foldername + "/conll/" + f
				
				conll = pd.read_csv(filename, sep = "\t", encoding="utf-8", header=None, names=conll_names)
				
				conll = conll >> mutate(
					SentenceNum = cumsum((X.TokenNum == 1).astype(int))
				) >> left_join(targets) 

				conll_sum = (conll >> 
					group_by(X.SentenceNum) >> 
					summarize(SentenceLength = n(X.SentenceNum))
				)	
				
				conll = (conll >> left_join(conll_sum) >>
					mutate(LastTokenFlag = (X.SentenceLength == X.TokenNum).astype(int))
				)
				
				# figure out which columns match the targets and if they also match the next
				for t in range(ntargets):
					conll['Target_'+str(t)] = (conll['Word'] == conll["_target_"+str(t)]).astype(int)
					if t == 0:
						conll['Target_Start'] = list(conll['Target_'+str(t)])	
					else:
						conll['Target_Start'] += list(conll['Target_'+str(t)] [t:]) + [0] * t	

				# keep only those matches in which the Target_Start value matches the number of targets
				for t in range(ntargets):
					next_vals = (conll['Target_Start'] == ntargets).astype(int)
					#print(next_vals[56:60])
					if t == 0:
						conll["Target_Keep"] = list(next_vals)						
					else:
						next_vals = [0] * t + list(next_vals)[:-t]
						conll["Target_Keep"] += next_vals
				
				# keep only those target matches that also have the keep flag
				for t in range(ntargets):
					conll['Target_'+str(t)] = conll['Target_'+str(t)] * conll['Target_Keep']
					conll = conll >> drop(X['_target_' + str(t)])

				conll = conll >> drop(X.Target_Keep, X.Target_Start)

				conll_sum = (conll >> group_by(X.SentenceNum) >> summarize(
						Sentence = concat(X.Word)
					))

				print(conll_sum[0:3])
				#print(conll[56:60])

				# iterate through each word in the conll gather sentence-level data
				#for index, row in conll.iterrows
				#	token_num = row['TokenNum']
					# do some stuff only onece, when we get a new sentence
				#	if token_num == 1:
				#		sentence_num = row['SentenceNum']
				#		sentence = ""
						# turn the targets at a current index into a list
				#		target_words = targets[sentence_num].split()

				#	sentence += 



				
		else:
			print("There are no conll and kwic files that match")
	else:
		print("you need conll and kwic sub-folders here with matching files")
else:
	print(foldername + "does not exist.")
