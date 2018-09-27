#coding: utf-8

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
from dfply import *


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
def _int(X):
	try:
		return [int(x) for x in list(X)]
	except:
		print("ERROR")
		print(X)

'''
BEGIN SCRIPT
'''

parser = argparse.ArgumentParser(description='Enter folder name that contains a data folder with the words.csv file.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a data folder with words.csv')

foldername = parser.parse_args().data_folder[0]

words_df = pd.read_csv(foldername + '/data/words.csv', encoding="utf-8")

# get just the targets of the words_df 
targets = (words_df >>
	select(contains('target_flag_'), ~X.target_flag)
)

ntargets = len(targets.columns)

passages = (words_df >> group_by(X.passage_num) >> summarize(
		passage_length = n(X.passage_num),
	)
)

for t in range(ntargets):
	passages >>= left_join(words_df >> group_by(X.passage_num) >> summarize(
		target_word_temp = concat_when(X.word, X['target_flag_' + str(t)], 1),
		target_lemma_temp = concat_when(X.lemma, X['target_flag_' + str(t)], 1),
		target_pos_temp = concat_when(X.pos, X['target_flag_' + str(t)], 1),
		target_cpos_temp = concat_when(X.cpos, X['target_flag_' + str(t)], 1),
	))
	passages = passages.rename(columns = { \
		'target_word_temp': 'target_' + str(t) + '_word', \
		'target_lemma_temp': 'target_' + str(t) + '_lemma', \
		'target_pos_temp': 'target_' + str(t) + '_pos', \
		'target_cpos_temp': 'target_' + str(t) + '_cpos', \
	})

# a set of features from words_df that will be applied to both targets
auto_features = ['num_children', 'num_distinct_child_cpos']
for feat in auto_features:
	for t in range(ntargets):
		passages >>= left_join(words_df >>
			group_by(X.passage_num) >> 
			summarize(
				_feat = concat_when(X[feat], X['target_flag_' + str(t)], 1)
			)
		)
		# make these features an int
		passages.loc[:,'_feat'] = passages.loc[:,'_feat'].astype(float).astype(int)
		passages = passages.rename(columns = {
			'_feat': 'target_'+ str(t) + '_' + feat, 
		})


# a curated, hand-crafted, artisanal, small-batch set of sentence-level features from words
passages >>= (
	left_join(words_df >> 
		group_by(X.passage_num) >> 
		summarize(
			target_0_num = concat_when(X['feature_num'], X['target_flag_0'], 1),
			target_0_ten = concat_when(X['feature_ten'], X['target_flag_0'], 1),
			target_0_per = concat_when(X['feature_per'], X['target_flag_0'], 1),
			target_1_num = concat_when(X['feature_num'], X['target_flag_1'], 1),
			target_1_gen = concat_when(X['feature_gen'], X['target_flag_1'], 1),
			#predictors
			#is the firs word an auxilary verb
			target_0_prev1_dependency_aux = concat_when(X['dfeature_prev1_dependency_aux'], X['target_flag_0'], 1),
		)
	) >> mutate(
		target_1_lemma_di = (X.target_1_lemma == 'di').astype(int),
		target_1_lemma_da = (X.target_1_lemma == 'da').astype(int),
		target_1_lemma_a = ((X.target_1_lemma == 'a') | (X.target_1_lemma == 'al')).astype(int),
		target_1_lemma_in = (X.target_1_lemma == 'in').astype(int),
	)
)



passages.to_csv(foldername + '/data/passages.csv', index=False, encoding='utf-8')
