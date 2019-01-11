#coding: utf-8

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from dfply import *

parser = argparse.ArgumentParser(description='Enter folder name that contains subfolders for conll and kwic data. The names \
	in these subfolders should match to ensure that target words get paired to the correct passages.')
parser.add_argument('data_folder', type=str,  action='store',
                   	help='a folder with subfolders for conll and kwic data')
parser.add_argument('kwic_pattern', type=str, nargs='?',
									  help='a regular expression pattern to find targets within the kwic file (include <>)')
parser.add_argument('--first-target', dest='use_first', action='store_const', const=1,
										help='use the first target if there are multiple (default last)')
parser.add_argument('--simple', dest='add_features', action='store_const', const=0,
										help='do not add additional features')
parser.add_argument('--drop-first-sentence', dest='drop_first_sentence', action='store_const', const=1,
										help='use this flag to drop the first sentence of a passage')

#print(parser.parse_args())
#sys.exit()

foldername = parser.parse_args().data_folder

if parser.parse_args().kwic_pattern is None:
	kwic_pattern = None
else:
	kwic_pattern = parser.parse_args().kwic_pattern

use_first = parser.parse_args().use_first

if parser.parse_args().add_features is None:
	add_features = 1
else:
	add_features = 0

drop_first_sentence = parser.parse_args().drop_first_sentence

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
					passage_nums = []
					i = 1
					for s in passages:
						if kwic_files is None:
							matches = re.search(r'\<(.*?)\>', s, re.IGNORECASE)
						else:
							matches = re.search(kwic_pattern, s, re.IGNORECASE)
						
						
						# print(s)
						# print(matches)
						if not matches is None:
							matches = matches.group(1).split()
							targets.append(matches)
							passage_nums.append(i)
						#if i==1818:
							#print(s)
							#print(matches)
							#print(targets)
							#sys.exit()
						i = i + 1
					

					ntargets = len(targets[0])
					target_names = ["target_" + str(t) for t in range(ntargets)]
					targets = pd.DataFrame(targets, columns=target_names)
					
					targets['passage_num'] = passage_nums #list(range(1,1+len(targets)))
					print('number of targets', targets['passage_num'].max())

					#print(targets)

				filename = foldername + "/data/conll/" + f
				# quoting=3 means no interpreting anything as quotes. This is important when a quote symbol is
				# the word and lemma (otherwise it thinks that there is text like " \t " instead of quote delimiter quote)
				words_df = pd.read_csv(filename, sep = "\t", encoding="utf-8", header=None, names=words_df_names,
					quoting=3)

				words_df >>= drop(X.X1, X.X2)

				# add a passage number field then join targets
				words_df >>= mutate(passage_num = cumsum((X.token_num == 1).astype(int))) >> left_join(targets)

				print('number of passages', words_df['passage_num'].max())

				#print(words_df.loc[words_df['passage_num']==1825])

				# for each passage count number of sentences (pos = "FS")
				words_df >>= (
					left_join(words_df >>
						group_by(X.passage_num) >>
						mutate(sentence_num = cumsum((X.pos == 'FS').astype(int))) >>
						mutate(sentence_num = lag(X.sentence_num + 1, 1))
					)
				)
				#words_df['sentence_num'] = pd.to_numeric(words_df['sentence_num'], downcast='integer')
				words_df = words_df.fillna(1, downcast='integer')

				# put passage_num and sentence_num in front
				cols = words_df.columns.tolist()
				cols.remove('passage_num')
				cols.remove('sentence_num')
				cols = ['passage_num', 'sentence_num'] + cols
				words_df = words_df[cols]

				if drop_first_sentence == 1:
					words_df >>= mask(X.sentence_num > 1)

				#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
				#	print(words_df[1:150])
				#sys.exit()

				# flag the last token of the passage
				words_df >>= (
					left_join(words_df >> 
						group_by(X.passage_num) >> 
						summarize(passage_length = n(X.passage_num))
					) >> mutate(
						last_token_flag = (X.passage_length == X.token_num).astype(int)
					) >> drop(X.passage_length)
				)

				# process targets
				
				# figure out which columns match the targets and if they also match the next
				for t in range(ntargets):
					words_df['_target_flag_'+str(t)] = (words_df['word'] == words_df["target_"+str(t)]).astype(int)
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
				
				# there may be multiple sets of the target, keep only the last (or first if flag set)
				words_df['target_flag'] = 0
				for t in range(ntargets):
					if use_first == 1:
						words_df >>= left_join(words_df >>
							group_by(X.passage_num) >>
							mask((X['_target_flag_'+str(t)] == 1) & (X._target_flag == 1)) >>
							summarize(
								token_num = first(X.token_num)							
							) >>
							mutate(
								is_token_at_position = 1
							)
						)
					else:
						words_df >>= left_join(words_df >>
							group_by(X.passage_num) >>
							mask((X['_target_flag_'+str(t)] == 1) & (X._target_flag == 1)) >>
							summarize(
								token_num = last(X.token_num)							
							) >>
							mutate(
								is_token_at_position = 1
							)
						)

					words_df.loc[:, 'is_token_at_position'] = np.where(words_df.loc[:, 'is_token_at_position'] == 1, 1, 0)
					words_df['target_flag_' + str(t)] = words_df['_target_flag_' + str(t)] * words_df.loc[:, 'is_token_at_position']
					words_df['target_flag'] = words_df['target_flag']  | words_df['target_flag_' + str(t)]
					# print(t)
					# print(words_df.columns)
					words_df >>= drop(['is_token_at_position', '_target_flag_' + str(t)])

				words_df >>= drop(['_target_flag', '_target_start'])
				'''	
				# keep only those target matches that also have the keep flag
				for t in range(ntargets):
					words_df['target_flag_'+str(t)] = words_df['target_flag_'+str(t)] * words_df['target_flag']
					words_df = words_df >> drop(X['_target_' + str(t)])
				'''
				



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

				if add_features == 1:
					# fill in missing values count values with 0's
					words_df >>= (
						mutate(
							num_children = np_where(X.num_children >= 0, X.num_children, 0),
							num_distinct_child_cpos = np_where(X.num_distinct_child_cpos >= 0, X.num_distinct_child_cpos, 0),
						)
					)

					# for each token get next and previous cpos
					words_df >>= (
						mutate(
							feature_cpos_next1 = lead(X.cpos),
							feature_cpos_prev1 = lag(X.cpos), 
						)
					)

					#### add other, ad-hoc important features
					words_df >>= (
						mutate(
							# features from words_df
							feature_gen = re_search_group(X.features, r'gen=(\w+)?\|', 1),
							feature_mod = re_search_group(X.features, r'mod=(\w+)?\|', 1),
							feature_num = re_search_group(X.features, r'num=(\w+)?\|', 1),
							feature_per = re_search_group(X.features, r'per=(\w+)?\|', 1),
							feature_ten = re_search_group(X.features, r'ten=(\w+)?\|', 1)
						) 
					)

					features_to_dummy = ['feature_gen', 'feature_mod', 'feature_num', 'feature_per', 'feature_ten', \
															 'feature_cpos_next1', 'feature_cpos_prev1']
					
					# turn categorical into dummy
					for feat in features_to_dummy:
						words_df = pd.concat([words_df, pd.get_dummies(words_df[feat], prefix='d' + feat)],
							axis=1, join_axes=[words_df.index])
					

				# words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_gen'], prefix='dfeature_gen')],
				# 		axis=1, join_axes=[words_df.index])
				# words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_mod'], prefix='dfeature_mod')],
				# 		axis=1, join_axes=[words_df.index])
				# words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_num'], prefix='dfeature_num')],
				# 		axis=1, join_axes=[words_df.index])
				# words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_per'], prefix='dfeature_per')],
				# 		axis=1, join_axes=[words_df.index])
				# words_df = pd.concat([words_df, pd.get_dummies(words_df['feature_ten'], prefix='dfeature_ten')],
				# 		axis=1, join_axes=[words_df.index])


				

				
				
				words_df.to_csv(foldername + '/data/words.csv', index=False, encoding='utf-8')
			
			with pd.option_context('display.max_rows', None, 'display.max_columns', None):
				print(words_df[90:120])

		else:
			print("There are no words_df and kwic files that match")
	else:
		print("you need words_df and kwic sub-folders here with matching files")
else:
	print(foldername + "does not exist.")
