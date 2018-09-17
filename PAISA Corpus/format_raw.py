#coding: utf-8

import os
import re
import argparse
import pandas as pd
import numpy as np
from dfply import *

parser = argparse.ArgumentParser(description='Enter folder name that contains subfolders for conll and kwic data. The names \
	in these subfolders should match to ensure that target words get paired to the correct passages.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder with subfolders for conll and kwic data')

foldername = parser.parse_args().data_folder[0]

@make_symbolic
def concat_when(words, filter_vals = None, keep_val = None, replace_with = None):
	''' the length of words and filter_vals must be the same '''
	words = list(words)
	if replace_with is None:
		replace_with = ''
	if filter_vals is None: 
		keep_val = True
	else:
		filter_vals = list(filter_vals)
	if filter_vals is None or len(words) == len(filter_vals):
		string = ''
		for i in range(len(words)):
			string += ' ' if i > 0 \
				and len(replace_with) > 0  \
				else ''
			if filter_vals is None or filter_vals[i] == keep_val:
				string += str(words[i])
			else:
				# are we substituting the target word here
				if re.search('{word}', replace_with):
					temp_word = re.sub('{word}', words[i], replace_with)
					string += temp_word
				else:
					string += replace_with
		string = re.sub(r"(\w)' +(\w)", r"\1'\2", string)
		string = re.sub(r' +([\.\?\])},;: ])', r'\1', string)
		string = re.sub(r'([(\[\{]) +', r'\1', string)
		string = re.sub(r'\s+', r' ', string)
		return string
	else:
		print('Error:' + str(len(words)) + ' word values, but ' + str(len(filter_vals)) + ' filter values.')
		return None

@make_symbolic
def re_search_group(series, pattern, group = None):
	out = []
	for s in series:
		n = re.search(pattern, s)
		if n: 
			if group:
				n = n.group(group)
			else:
				n = n.group(0)
		else:
			n = ''
		out.append(n)
	return out

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
					passages = in_file.readlines()
					# get passage w/o target and target
					targets = []
					target_names = []
					for s in passages:
						matches = re.search(r'(.*)?<(.*?)>(.*)', s)
						matches = matches.group(2).split()
						targets.append(matches)
					
					ntargets = len(targets[0])
					target_names = ["_target_" + str(t) for t in range(ntargets)]
					targets = pd.DataFrame(targets, columns=target_names)
					targets['PassageNum'] = list(range(1,1+len(targets)))
				filename = foldername + "/data/conll/" + f
				
				conll = pd.read_csv(filename, sep = "\t", encoding="utf-8", header=None, names=conll_names)
				
				conll >>= mutate(
					PassageNum = cumsum((X.TokenNum == 1).astype(int)),
					# Features from conll
					Feature_gen = re_search_group(X.Features, r'gen=(\w+)?\|', 1),
					Feature_mod = re_search_group(X.Features, r'mod=(\w+)?\|', 1),
					Feature_num = re_search_group(X.Features, r'num=(\w+)?\|', 1),
					Feature_per = re_search_group(X.Features, r'per=(\w+)?\|', 1),
					Feature_ten = re_search_group(X.Features, r'ten=(\w+)?\|', 1)
				) >> left_join(targets) 

				# put PassageNum in front
				cols = conll.columns.tolist()
				cols.remove('PassageNum')
				cols = ['PassageNum'] + cols
				conll = conll[cols]
				
				conll >>= left_join(conll >> 
					group_by(X.PassageNum) >> 
					summarize(PassageLength = n(X.PassageNum))
				) >> mutate(
					LastTokenFlag = (X.PassageLength == X.TokenNum).astype(int)
				) >> drop(X.PassageLength)
				
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

				conll >>= drop(X._target_start, X.X1, X.X2)

				print(conll[80:90])

				# prepare the targets data frame for binding, put all targets together
				targets['Target'] = ''
				for t in range(ntargets):
					targets['Target'] += ' ' if t > 0 else ''
					targets['Target'] += targets['_target_' + str(t)]
					targets = targets >> drop(X['_target_' + str(t)])

				passages = (conll >> group_by(X.PassageNum) >> summarize(
						PassageLength = n(X.PassageNum),

				) >> ungroup >> left_join(targets))

				for t in range(ntargets):
					passages >>= left_join(conll >> group_by(X.PassageNum) >> summarize(
						Target_Word_temp = concat_when(X.Word, X['Target_' + str(t)], 1),
						Target_Lemma_temp = concat_when(X.Lemma, X['Target_' + str(t)], 1),
						Target_POS_temp = concat_when(X.POS, X['Target_' + str(t)], 1),
						Target_CPOS_temp = concat_when(X.CPOS, X['Target_' + str(t)], 1),
					))
					passages = passages.rename(columns = { \
						'Target_Word_temp': 'Target_Word_' + str(t), \
						'Target_Lemma_temp': 'Target_Lemma_' + str(t), \
						'Target_POS_temp': 'Target_POS_' + str(t), \
						'Target_CPOS_temp': 'Target_CPOS_' + str(t), \
					})

				passages.to_csv(foldername + '/data/passages.csv', index=False, encoding='utf-8')
				conll.to_csv(foldername + '/data/words.csv', index=False, encoding='utf-8')
				
		else:
			print("There are no conll and kwic files that match")
	else:
		print("you need conll and kwic sub-folders here with matching files")
else:
	print(foldername + "does not exist.")
