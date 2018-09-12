#coding: utf-8

import argparse
import pandas as pd
import numpy.random as rd
import re

parser = argparse.ArgumentParser(description='Enter a processed "sentences_[my file].csv" file to play a generation game.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder containing a sentences file with the following columns: Sentence, Target, Sentence_no_target')

foldername = parser.parse_args().data_folder[0]

sentences = pd.read_csv(foldername + '/data/sentences.csv', encoding="utf-8")

while (True):
	randi = rd.randint(0, len(sentences.index)-1)
	sentence_no_target = sentences['Sentence_No_Target'].iloc[randi]
	sentence = sentences['Sentence'].iloc[randi]
	target = sentences['Target'].iloc[randi]
	print("****************************************************************")
	print(sentence_no_target)
	x = input('\nInserisci le parole corrette per ?___? \n0 per uscire\n\n')
	if (x == '0'):
		break;
	elif (x == target):
		print("corretto!")
		print('  _________\n /         \\\n |  O   0  |\n |    -    |\n |  \\___/  |\n \\_________/')
	else:
		print("Spiacente")
		print('  _________\n /         \\\n |  /\\ /\\  |\n |    -    |\n |   ___   |\n |  /   \\  |\n \\_________/');
		print(sentence)
	print("\n")
	#print(sentences[randi])
