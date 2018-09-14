#coding: utf-8

import argparse
import re
import pandas as pd
import numpy.random as rd
from dfply import *
from clint.textui import puts, indent, colored, prompt, validators
from wiktionaryparser import WiktionaryParser

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
		string = re.sub(r"(\w)' +(\w)", r"\1'\2", string)
		string = re.sub(r' +([\.\?\])},;: ])', r'\1', string)
		string = re.sub(r'([(\[\{]) +', r'\1', string)
		return string
	else:
		print('Error:' + str(len(words)) + ' word values, but ' + str(len(filter_vals)) + ' filter values.')
		return None


arg_parser = argparse.ArgumentParser(description='Enter a processed "sentences_[my file].csv" file to play a generation game.')
arg_parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder containing a sentences file with the following columns: Sentence, Target, Sentence_no_target')

def_parser = WiktionaryParser()
def_parser.set_default_language('italian')

foldername = arg_parser.parse_args().data_folder[0]

sentence_df = pd.read_csv(foldername + '/data/sentences.csv', encoding="utf-8")
word_df = pd.read_csv(foldername + '/data/words.csv', encoding="utf-8")

total_points = 0

#program loop
while True:
	randi = rd.randint(0, len(sentence_df.index)-1)
	sentence_row = sentence_df.iloc[randi]
	sentence_num = sentence_row['SentenceNum']
	sentence_word_df = word_df >> mask(X.SentenceNum == sentence_num)
	sentence_no_target = concat_when(sentence_word_df['Word'], sentence_word_df['TargetFlag'], 0, '*___*')		
	target = sentence_row['Target']
	puts(colored.black('****************************************************************'))
	
	# sentence loop
	while True:
		break_all = False
		puts(colored.black(sentence_no_target))
		puts(colored.black('\nInserisci le parole corrette per *___*. (e` -> è) \
			\n?parole per aiuto	\
			\n0 per uscire\n'))
		with indent(4, quote=' >'):
			user_words = prompt.query('>>> ')
		# replace o` with ò, replace e` with è, a` with à
		user_words = re.sub(r'o`', 'ò', user_words)
		user_words = re.sub(r'e`', 'è', user_words)
		user_words = re.sub(r'a`', 'à', user_words)
		user_words = re.sub(r'u`', 'ù', user_words)

		if user_words == '0':
			break_all = True
			break;
		elif user_words[0] == "?":
			print(def_parser.fetch(user_words[1:]))
			print('\n')
		else:
			points = 0
			user_words = user_words.split()
			target_word_df = sentence_word_df >> mask(X.TargetFlag == 1)
			for i in range(len(user_words)):
				user_word = user_words[i]
				target_word = target_word_df['Word'].iloc[i]
				target_lemma = target_word_df['Lemma'].iloc[i]
				target_cpos = target_word_df['CPOS'].iloc[i]
				target_lemma_cpos = target_lemma + "|" + target_cpos
				# get all instances of the words df that match the user's word, used to find lemma-cpos
				lemma_cposs = list(word_df >> mask(X.Word == user_word) >> 
					distinct(X.Word, X.Lemma, X.CPOS) >>
					mutate(Lemma_CPOS = X.Lemma + "|" + X.CPOS) >>
					pull('Lemma_CPOS')
				)
				if user_word == target_word:
					points += 3
				elif target_lemma_cpos in lemma_cposs:
					points += 1
			total_points += points
			if points > 2:
				puts(colored.cyan(' Ben fatto, ' + str(points) + " punti."))
				puts(colored.cyan('  _________\n /         \\\n |  O   O  |\n |    J    |\n |  \\___/  |\n \\_________/'))
			else:
				puts(colored.red(' Spiacente, solo ' + str(points) + " punti."))
				puts(colored.red('  _________\n /         \\\n |  /\\ /\\  |\n |    J    |\n |   ___   |\n |  /   \\  |\n \\_________/'))
			# break out of the sentence loop
			break
	if break_all:
		break
	sentence = concat_when(sentence_word_df['Word'], sentence_word_df['TargetFlag'], 0, '*{word}*')		
	puts(colored.black(sentence))
	puts("\n")
	puts(colored.magenta('(Ha '+ str(total_points) + ' punti.)'))