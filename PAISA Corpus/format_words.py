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
def nchar(words):
	return [len(x) for x in list(words)]


@make_symbolic
def np_where(bools, val_if_true, val_if_false):
	return list(np.where(bools, val_if_true, val_if_false))


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

words_df_names = ['token_num', "word", "lemma", "cpos", "pos", "features", "parent", "dependency", "X1", "X2"]

if os.path.isdir(foldername):
	# we should have matching files in a conll and a kwic directory, make sure they are both there
	if "conll" in os.listdir(foldername + "/data") and "kwic" in os.listdir(foldername + "/data"):
		words_df_files = os.listdir(foldername + "/data/conll")
		kwic_files = os.listdir(foldername + "/data/kwic")
		match_files = list(set(words_df_files) & set(kwic_files))
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
				words_df = pd.read_csv(filename, sep = "\t", encoding="utf-8", header=None, names=words_df_names,
					quoting=3)
				
				words_df >>= (
					drop(X.X1, X.X2) >>
					mutate(
						passage_num = cumsum((X.token_num == 1).astype(int)),
						# features from words_df
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
				words_df >>= (
					left_join(words_df >> (
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
				# fill in missing values count values with 0's
				words_df >>= (
					mutate(
						num_children = np_where(X.num_children >= 0, X.num_children, 0),
						num_distinct_child_cpos = np_where(X.num_distinct_child_cpos >= 0, X.num_distinct_child_cpos, 0),
					)
				)
				
				# turn categorical into dummy
				words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_gen'], prefix='dfeature_gen')],
						axis=1, join_axes=[words_df.index])
				words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_mod'], prefix='dfeature_mod')],
						axis=1, join_axes=[words_df.index])
				words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_num'], prefix='dfeature_num')],
						axis=1, join_axes=[words_df.index])
				words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_per'], prefix='dfeature_per')],
						axis=1, join_axes=[words_df.index])
				words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_ten'], prefix='dfeature_ten')],
						axis=1, join_axes=[words_df.index])

				# ad-hoc dummies
				words_df >>= (mutate(
						dfeature_dependency_parent = (X.dependency == 'parent').astype(int),					
						dfeature_dependency_aux = (X.dependency == 'aux').astype(int),
						dfeature_prev1_dependency_aux = (lag(X.dependency, 1) == 'aux').astype(int),
					)
				)

				# get the list of dummy features, we will need this for our learning model
				dummy_features = list(filter(re.compile('dfeature').search, words_df.columns.tolist()))				
				
				words_df >>= (
					left_join(words_df >> 
						group_by(X.passage_num) >> 
						summarize(passage_length = n(X.passage_num))
					) >> mutate(
						last_token_flag = (X.passage_length == X.token_num).astype(int)
					) >> drop(X.passage_length)
				)

				# figure out which columns match the targets and if they also match the next
				for t in range(ntargets):
					words_df['_target_flag_'+str(t)] = (words_df['word'] == words_df["_target_"+str(t)]).astype(int)
					if t == 0:
						words_df['_target_start'] = list(words_df['_target_flag_'+str(t)])	
					else:
						words_df['_target_start'] += list(words_df['_target_flag_'+str(t)] [t:]) + [0] * t	

				# keep only those matches in which the _target_start value matches the number of targets
				for t in range(ntargets):
					next_vals = (words_df['_target_start'] == ntargets).astype(int)
					#print(next_vals[56:60])
					if t == 0:
						words_df["_target_flag"] = list(next_vals)						
					else:
						next_vals = [0] * t + list(next_vals)[:-t]
						words_df["_target_flag"] += next_vals
				
				# there may be multiple sets of the target, keep only the first
				words_df['target_flag'] = 0
				for t in range(ntargets):
					words_df >>= left_join(words_df >>
						group_by(X.passage_num) >>
						mask((X['_target_flag_'+str(t)] == 1) & (X._target_flag == 1)) >>
						summarize(
							token_num = last(X.token_num),
							
						) >>
						mutate(
							is_final_token = 1
						)
					)

					words_df.loc[:, 'is_final_token'] = np.where(words_df.loc[:, 'is_final_token'] == 1, 1, 0)
					words_df['target_flag_' + str(t)] = words_df['_target_flag_' + str(t)] * words_df.loc[:, 'is_final_token']
					words_df['target_flag'] = words_df['target_flag']  | words_df['target_flag_' + str(t)]
					print(t)
					print(words_df.columns)
					words_df >>= drop(['is_final_token', '_target_flag_' + str(t)])

				words_df >>= drop(['_target_flag', '_target_start'])
				'''	
				# keep only those target matches that also have the keep flag
				for t in range(ntargets):
					words_df['target_flag_'+str(t)] = words_df['target_flag_'+str(t)] * words_df['target_flag']
					words_df = words_df >> drop(X['_target_' + str(t)])
				'''
				

				# put passage_num in front
				cols = words_df.columns.tolist()
				cols.remove('passage_num')
				cols = ['passage_num'] + cols
				words_df = words_df[cols]
				
				words_df.to_csv(foldername + '/data/words.csv', index=False, encoding='utf-8')
			
			with pd.option_context('display.max_rows', None, 'display.max_columns', None):
				print(words_df[80:90])

		else:
			print("There are no words_df and kwic files that match")
	else:
		print("you need words_df and kwic sub-folders here with matching files")
else:
	print(foldername + "does not exist.")
