#coding: utf-8

import os
import re
import argparse
import pandas as pd
import numpy as np
from dfply import *

parser = argparse.ArgumentParser(description='Enter folder name that contains subfolders for conll and kwic data. The names \
	in these subfolders should match to ensure that target words get paired to the correct sentences.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder with subfolders for conll and kwic data')

foldername = parser.parse_args().data_folder[0]

@make_symbolic
def concat(words):
	string = ' '.join(str(x) for x in words)
	string = re.sub(r"(\w)' (\w)", r"\1'\2", string)
	string = re.sub(r' ([\.\?\)\]\},;: ])', r'\1', string)
	string = re.sub(r'([\(\[\{]) ', r'\1', string)
	return string

@make_symbolic
def concat_when(words, filters, keep_val, replace_with):
	''' the length of words and filters must be the same '''
	words = list(words)
	filters = list(filters)
	if len(words) == len(filters):
		string = ''
		for i in range(len(words)):
			string += ' ' if i > 0 else ''
			if filters[i] == keep_val:
				string += str(words[i])
			else:
				string += 'REPLACEMEAFTERRESUBSHOPEFULLYTHISISNTINTHETEXT'
		string = re.sub(r"(\w)' (\w)", r"\1'\2", string)
		string = re.sub(r' ([\.\?\)\]\},;: ])', r'\1', string)
		string = re.sub(r'([\(\[\{]) ', r'\1', string)
		string = re.sub('REPLACEMEAFTERRESUBSHOPEFULLYTHISISNTINTHETEXT', replace_with, string)
		return string
	else:
		print('Error:' + str(len(words)) + ' word values, but ' + str(len(filters)) + ' filter values.')
		return None

conll_names = ['TokenNum', "Word", "Lemma", "CPOS", "POS", "Features", "Head", "Dependency", "X1", "X2"]

if os.path.isdir(foldername):
	# we should have matching files in a conll and a kwic directory, make sure they are both there
	if "conll" in os.listdir(foldername + "/data") and "kwic" in os.listdir(foldername + "/data"):
		conll_files = os.listdir(foldername + "/data/conll")
		kwic_files = os.listdir(foldername + "/data/kwic")
		match_files = list(set(conll_files) & set(kwic_files))
		if len(match_files) > 0:
			for f in match_files:
				filename = foldername + "/data/kwic/" + f
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
				filename = foldername + "/data/conll/" + f
				
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
						conll['_target_start'] = list(conll['Target_'+str(t)])	
					else:
						conll['_target_start'] += list(conll['Target_'+str(t)] [t:]) + [0] * t	

				# keep only those matches in which the _target_start value matches the number of targets
				for t in range(ntargets):
					next_vals = (conll['_target_start'] == ntargets).astype(int)
					#print(next_vals[56:60])
					if t == 0:
						conll["TargetFlag"] = list(next_vals)						
					else:
						next_vals = [0] * t + list(next_vals)[:-t]
						conll["TargetFlag"] += next_vals
				
				# keep only those target matches that also have the keep flag
				for t in range(ntargets):
					conll['Target_'+str(t)] = conll['Target_'+str(t)] * conll['TargetFlag']
					conll = conll >> drop(X['_target_' + str(t)])

				conll = conll >> drop(X._target_start, X.X1, X.X2)

				print(conll[80:90])

				# prepare the targets data frame for binding, put all targets together
				targets['Target'] = ''
				for t in range(ntargets):
					targets['Target'] += ' ' if t > 0 else ''
					targets['Target'] += targets['_target_' + str(t)]
					targets = targets >> drop(X['_target_' + str(t)])

				sentences = (conll >> group_by(X.SentenceNum) >> summarize(
						Sentence = concat(X.Word),
						Sentence_No_Target = concat_when(X.Word, X.TargetFlag, 0, '?___?')
					) >> ungroup >> left_join(targets))

				#print(list(sentences.Sentence)[0:3])
				
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

				sentences.to_csv(foldername + '/data/sentences.csv', index=False, encoding='utf-8')
				conll.to_csv(foldername + '/data/words.csv', index=False, encoding='utf-8')
				
		else:
			print("There are no conll and kwic files that match")
	else:
		print("you need conll and kwic sub-folders here with matching files")
else:
	print(foldername + "does not exist.")
