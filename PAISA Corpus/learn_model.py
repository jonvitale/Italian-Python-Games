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
parser.add_argument('-m',type=str, default='nb',
                   help='a learning algorithm: nb (naive bayes), tree (decision tree)')

foldername = parser.parse_args().data_folder[0]
method = parser.parse_args().m

print(parser.parse_args())
print(parser.parse_args().m)

passages_df = pd.read_csv(foldername + '/data/passages.csv', encoding="utf-8")

target_name = 'target_1_lemma'
target = passages_df[target_name]

predictors = passages_df.filter(regex='child|depend|target_1.*?next1')

print('target:', target_name)
print('predictors:')
print(predictors.dtypes)

if method == 'nb':
	from sklearn.naive_bayes import BernoulliNB


	mnb = BernoulliNB()
	fit = mnb.fit(predictors, target)
	y_pred = fit.predict(predictors)
	

elif method == 'tree':
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	fit = clf.fit(predictors, target)
	y_pred = fit.predict(predictors)

	import graphviz
	dot_data = tree.export_graphviz(fit, out_file=None)
	graph = graphviz.Source(dot_data)
	graph
	#graph.render(foldername + "_dtree.pdf")

print("Number of mislabeled points out of a total %d points : %d"
       % (predictors.shape[0],(target != y_pred).sum()))