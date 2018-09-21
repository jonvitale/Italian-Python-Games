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
def concat_when(words, filter_vals = None, keep_val = None, replace_with = None, sep=' '):
	''' This function concats strings from the words list on the occasion
			that the value in filter_vals matches the value in keep_val,
			if there is not a match then substitue with replace_with, unless None, then
			simply do not add anything for the given index
	'''
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
			string += sep if i > 0 \
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

conll_names = ['token_num', "word", "lemma", "cpos", "pos", "features", "parent", "dependency", "X1", "X2"]

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
					targets['passage_num'] = list(range(1,1+len(targets)))

				filename = foldername + "/data/conll/" + f
				# quoting=3 means no interpreting anything as quotes. This is important when a quote symbol is
				# the word and lemma (otherwise it thinks that there is text like " \t " instead of quote delimiter quote)
				conll = pd.read_csv(filename, sep = "\t", encoding="utf-8", header=None, names=conll_names,
					quoting=3)
				
				conll >>= (
					drop(X.X1, X.X2) >>
					mutate(
						passage_num = cumsum((X.token_num == 1).astype(int)),
						# features from conll
						feature_gen = re_search_group(X.features, r'gen=(\w+)?\|', 1),
						feature_mod = re_search_group(X.features, r'mod=(\w+)?\|', 1),
						feature_num = re_search_group(X.features, r'num=(\w+)?\|', 1),
						feature_per = re_search_group(X.features, r'per=(\w+)?\|', 1),
						feature_ten = re_search_group(X.features, r'ten=(\w+)?\|', 1)
					) >>
					# add each and all targets 
					left_join(targets)
				)

				# For each head, get all dependencies in a list (| separated)
				conll >>= (
					left_join(conll >> (
							mutate(
								parent = X.parent.astype(int),
								token_num = X.token_num.astype(str)
							) >>
							mask(X.parent != 0) >>
							group_by(X.passage_num, X.parent) >>
							summarize(
								children = X.token_num.str.cat(sep = '|'),
								num_children = n_distinct(X.token_num),
								num_distinct_child_cpos = n_distinct(X.cpos)
							) >>
							select(X.passage_num, X.parent, X.children, X.num_children, X.num_distinct_child_cpos) >>
							rename(token_num = X.parent)
						)
					)
				)
				
				# turn categorical into dummy
				conll = pd.concat([conll, pd.get_dummies(conll['feature_gen'], prefix='dfeature_gen')],
						axis=1, join_axes=[conll.index])
				conll = pd.concat([conll, pd.get_dummies(conll['feature_mod'], prefix='dfeature_mod')],
						axis=1, join_axes=[conll.index])
				conll = pd.concat([conll, pd.get_dummies(conll['feature_num'], prefix='dfeature_num')],
						axis=1, join_axes=[conll.index])
				conll = pd.concat([conll, pd.get_dummies(conll['feature_per'], prefix='dfeature_per')],
						axis=1, join_axes=[conll.index])
				conll = pd.concat([conll, pd.get_dummies(conll['feature_ten'], prefix='dfeature_ten')],
						axis=1, join_axes=[conll.index])

				# ad-hoc dummies
				conll >>= (mutate(
						dfeature_dependency_ROOT = (X.dependency == 'ROOT').astype(int),					
						dfeature_dependency_aux = (X.dependency == 'aux').astype(int),
						dfeature_prev1_dependency_aux = (lag(X.dependency, 1) == 'aux').astype(int),
					)
				)

				# get the list of dummy features, we will need this for our learning model
				dummy_features = list(filter(re.compile('dfeature').search, conll.columns.tolist()))				
				
				conll >>= (
					left_join(conll >> 
						group_by(X.passage_num) >> 
						summarize(passage_length = n(X.passage_num))
					) >> mutate(
						last_token_flag = (X.passage_length == X.token_num).astype(int)
					) >> drop(X.passage_length)
				)

				# figure out which columns match the targets and if they also match the next
				for t in range(ntargets):
					conll['target_'+str(t)] = (conll['word'] == conll["_target_"+str(t)]).astype(int)
					if t == 0:
						conll['_target_start'] = list(conll['target_'+str(t)])	
					else:
						conll['_target_start'] += list(conll['target_'+str(t)] [t:]) + [0] * t	

				# keep only those matches in which the _target_start value matches the number of targets
				for t in range(ntargets):
					next_vals = (conll['_target_start'] == ntargets).astype(int)
					#print(next_vals[56:60])
					if t == 0:
						conll["targetFlag"] = list(next_vals)						
					else:
						next_vals = [0] * t + list(next_vals)[:-t]
						conll["targetFlag"] += next_vals
				
				# keep only those target matches that also have the keep flag
				for t in range(ntargets):
					conll['target_'+str(t)] = conll['target_'+str(t)] * conll['targetFlag']
					conll = conll >> drop(X['_target_' + str(t)])

				# finalize the conll dataframe (remove unneceasry)
				conll >>= drop(X._target_start)
				# put passage_num in front
				cols = conll.columns.tolist()
				cols.remove('passage_num')
				cols = ['passage_num'] + cols
				conll = conll[cols]
				
				
				# prepare the targets data frame for binding, put all targets together
				targets['target'] = ''
				for t in range(ntargets):
					targets['target'] += ' ' if t > 0 else ''
					targets['target'] += targets['_target_' + str(t)]
					targets = targets >> drop(X['_target_' + str(t)])

				passages = (conll >> group_by(X.passage_num) >> summarize(
						passage_length = n(X.passage_num),

				) >> ungroup >> left_join(targets))

				for t in range(ntargets):
					passages >>= left_join(conll >> group_by(X.passage_num) >> summarize(
						target_word_temp = concat_when(X.word, X['target_' + str(t)], 1),
						target_lemma_temp = concat_when(X.lemma, X['target_' + str(t)], 1),
						target_pos_temp = concat_when(X.pos, X['target_' + str(t)], 1),
						target_cpos_temp = concat_when(X.cpos, X['target_' + str(t)], 1),
					))
					passages = passages.rename(columns = { \
						'target_word_temp': 'target_' + str(t) + '_word', \
						'target_lemma_temp': 'target_' + str(t) + '_lemma', \
						'target_pos_temp': 'target_' + str(t) + '_pos', \
						'target_cpos_temp': 'target_' + str(t) + '_cpos', \
					})
					# add numerical features (dummy)
					#for dummy in dummy_features:
					#	passages >>= left_join(conll >> group_by(X.passage_num) >> summarize(
					#		Dummy = concat_when(X[dummy], X['target_' + str(t)], 1)
					#	))
					#	passages = passages.rename(columns = { \
					#		'Dummy': 'target_'+ str(t) + '_' + dummy, 
					#	})

				# a curated, hand-crafted, artisanal, small-batch set of sentence-level features
				passages >>= left_join(conll >> group_by(X.passage_num) >> summarize(
					target_0_num = concat_when(X['feature_num'], X['target_0'], 1),
					target_0_ten = concat_when(X['feature_ten'], X['target_0'], 1),
					target_0_per = concat_when(X['feature_per'], X['target_0'], 1),
					target_1_num = concat_when(X['feature_num'], X['target_1'], 1),
					target_1_gen = concat_when(X['feature_gen'], X['target_1'], 1),
					#predictors
					#is the firs word an auxilary verb
					target_0_prev1_dependency_aux = concat_when(X['dfeature_prev1_dependency_aux'], X['target_0'], 1),
				))

				passages.to_csv(foldername + '/data/passages.csv', index=False, encoding='utf-8')
				conll.to_csv(foldername + '/data/words.csv', index=False, encoding='utf-8')
			
			with pd.option_context('display.max_rows', None, 'display.max_columns', None):
				print(conll[80:90])

		else:
			print("There are no conll and kwic files that match")
	else:
		print("you need conll and kwic sub-folders here with matching files")
else:
	print(foldername + "does not exist.")
