#coding: utf-8

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
#from dfply import *
from sklearn import cluster

parser = argparse.ArgumentParser(description='Enter folder name that contains a data folder with the passages.csv file.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a data folder with passages.csv')

foldername = parser.parse_args().data_folder[0]

passages_df = pd.read_csv(foldername + '/data/passages.csv', encoding="utf-8")

# default naive bayes
method = 'nb'

if method == 'nb':
	from sklearn.naive_bayes import MultinomialNB

	target_name = 'target_1_lemma_di'
	target = passages_df[target_name]

	predictors = passages_df.filter(regex='child|prev')

	print('target:', target_name)
	print('predictors:', predictors.columns)
	print(predictors.dtypes)

	mnb = MultinomialNB()
	y_pred = mnb.fit(predictors, target).predict(predictors)
	print("Number of mislabeled points out of a total %d points : %d"
       % (predictors.shape[0],(target != y_pred).sum()))