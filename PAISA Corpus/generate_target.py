import argparse
import pandas as pd
import numpy.random as rd
import re

parser = argparse.ArgumentParser(description='Enter a processed "sentences_[my file].csv" file to play a generation game.')
parser.add_argument('sentences_file', metavar='N', type=str, nargs='+',
                   help='a sentences file with the following columsn: Sentence, Target, Sentence_no_target')

filename = parser.parse_args().sentences_file[0]

sentences = pd.read_csv(filename, encoding="latin1")

while (True):
	randi = rd.randint(0, len(sentences.index)-1)
	sentence_no_target = sentences['Sentence_no_target'].iloc[randi]
	sentence = sentences['Sentence'].iloc[randi]
	target = sentences['Target'].iloc[randi]
	print("****************************************************************")
	print(sentence_no_target)
	x = input('inserisci le parole corrette per ?___? (o 0 per uscire) \n')
	if (x == '0'):
		break;
	elif (x == target):
		print("corretto! :-)")
	else:
		print("Spiacente :-(")
		print(sentence)
	print("\n")
	#print(sentences[randi])
