#coding: utf-8

import argparse
import re
import pandas as pd
import numpy.random as rd
from dfply import *
from clint.textui import puts, indent, colored, prompt, validators


@make_symbolic
def concat_when(words, filter_vals, keep_val, replace_with):
	''' the length of words and filter_vals must be the same '''
	words = list(words)
	filter_vals = list(filter_vals)
	if len(words) == len(filter_vals):
		string = ''
		for i in range(len(words)):
			string += ' ' if i > 0 else ''
			if filter_vals[i] == keep_val:
				string += str(words[i])
			else:
				# are we substituting the target word here
				if re.search('{word}', replace_with):
					temp_word = re.sub('{word}', words[i], replace_with)
					string += temp_word
				else:
					string += replace_with
		string = re.sub(r"(\w)' (\w)", r"\1'\2", string)
		string = re.sub(r' ([\.\?\])},;: ])', r'\1', string)
		string = re.sub(r'([(\[\{]) ', r'\1', string)
		#string = re.sub('ReplaceMeAfterRESubs', replace_with, string)
		return string
	else:
		print('Error:' + str(len(words)) + ' word values, but ' + str(len(filter_vals)) + ' filter values.')
		return None


parser = argparse.ArgumentParser(description='Enter a processed "sentences_[my file].csv" file to play a generation game.')
parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder containing a sentences file with the following columns: Sentence, Target, Sentence_no_target')

foldername = parser.parse_args().data_folder[0]

sentences = pd.read_csv(foldername + '/data/sentences.csv', encoding="utf-8")
words = pd.read_csv(foldername + '/data/words.csv', encoding="utf-8")

total_points = 0

while (True):
	randi = rd.randint(0, len(sentences.index)-1)
	sentence_row = sentences.iloc[randi]
	sentence_num = sentence_row['SentenceNum']
	sentence_words = words >> mask(X.SentenceNum == sentence_num)
	sentence_no_target = concat_when(sentence_words['Word'], sentence_words['TargetFlag'], 0, '*___*')		
	target = sentence_row['Target']
	puts(colored.black('****************************************************************'))
	puts(colored.black(sentence_no_target))
	puts(colored.black('\nInserisci le parole corrette per *___*. (e` -> è)\n0 per uscire\n'))
	with indent(4, quote=' >'):
		#x = input('')
		x = prompt.query('>>> ')
	# replace o` with ò, replace e` with è, a` with à
	x = re.sub(r'o`', 'ò', x)
	x = re.sub(r'e`', 'è', x)
	x = re.sub(r'a`', 'à', x)
	x = re.sub(r'u`', 'ù', x)

	if (x == '0'):
		break;
	else:
		points = 0
		x = x.split()
		target_words = sentence_words >> mask(X.TargetFlag == 1)
		for i in range(len(x)):
			if x[i] == target_words['Word'].iloc[i]:
				points += 3
			elif x[i] == target_words['Lemma'].iloc[i]:
				points += 1
		total_points += points
		if points > 2:
			puts(colored.cyan(' Fatto benne, ' + str(points) + " punti."))
			puts(colored.cyan('  _________\n /         \\\n |  O   O  |\n |    J    |\n |  \\___/  |\n \\_________/'))
		else:
			puts(colored.red(' Spiacente, solo ' + str(points) + " punti."))
			puts(colored.red('  _________\n /         \\\n |  /\\ /\\  |\n |    J    |\n |   ___   |\n |  /   \\  |\n \\_________/'))
	
	sentence = concat_when(sentence_words['Word'], sentence_words['TargetFlag'], 0, '*{word}*')		
	puts(colored.black(sentence))
	puts("\n")
	puts(colored.magenta('(Ha '+ str(total_points) + ' punti.)'))