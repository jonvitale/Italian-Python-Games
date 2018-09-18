#coding: utf-8

import argparse
import re
import pandas as pd
import numpy.random as rd
from dfply import *
from clint.textui import puts, indent, colored, prompt, validators
from wiktionaryparser import WiktionaryParser

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
def nchar(words):
	return [len(x) for x in list(words)]

@make_symbolic
def if_else(bools, val_if_true, val_if_false):
	out = []
	for b in bools:
		out.append(val_if_true if b else val_if_false)
	return out

@make_symbolic
def if_match_else(vals, targets, vals_if_true, vals_if_false):
	vals = list(vals)
	if isinstance(targets, pd.core.series.Series) or isinstance(targets, list):
		targets = list(targets)
	else:
		targets = [targets for i in range(len(vals))]

	if isinstance(vals_if_true, pd.core.series.Series) or isinstance(vals_if_true, list):
		vals_if_true = list(vals_if_true)
	else:
		vals_if_true = [vals_if_true for i in range(len(vals))]

	if isinstance(vals_if_false, pd.core.series.Series) or isinstance(vals_if_false, list):
		vals_if_false = list(vals_if_false)
	else:
		vals_if_false = [vals_if_false for i in range(len(vals))]
	
	out = []
	for i in range(len(vals)):
		#print(vals[i] + " - " + targets[i])	
		if vals[i] == targets[i]:
			out.append(vals_if_true[i])
		else:
			out.append(vals_if_false[i])
	return out

def pprint_wiktionary(word):
	#try:
	flag = re.search(r' \-(\w+)', word)
	if flag:
		flag = flag.group(1)
	word = re.sub(r' \-(\w+)', '', word)
	entry = def_parser.fetch(word)
	if flag == 'debug' or flag == 'd':
		print(entry)
	# an entry may have several definitions (senses)
	for i in range(len(entry)):
		print('----------------------------------------------------')
		print(' ' + word + ' ' + str(i+1) + ':')
		if entry[i]['etymology'] and len(entry[i]['etymology']) > 0:
			print(' --- Etymology ---')
			print(entry[i]['etymology'])
		if entry[i]['definitions'] and len(entry[i]['definitions']) > 0:
			for j in range(len(entry[i]['definitions'])):
				definition = entry[i]['definitions'][j]
				if definition['partOfSpeech'] and len(definition['partOfSpeech']) > 0:
					print(' --- Part Of Speech ---')
					print(definition['partOfSpeech'])
				if definition['text'] and len(definition['text']) > 0:
					print(' --- Definition ---')
					for k in range(len(definition['text'])):
						print(str(k+1) + '. ' + definition['text'][k])	
				if definition['relatedWords'] and len(definition['relatedWords']) > 0:
					for m in range(len(definition['relatedWords'])):
						related = definition['relatedWords'][m]
						if related['relationshipType'] and len(related['words']) > 0:
							print(' --- ' + related['relationshipType'] + " " + str(m+1) + '---')
							print(related['words'])	
				if definition['examples'] and len(definition['examples']) > 0:
					print(' --- Examples ---')
					for m in range(len(definition['examples'])):
						print(definition['examples'][m])	
		print('----------------------------------------------------')
		print('\n')					
	#except:
	#	print("couldn't access Wiktionary")

arg_parser = argparse.ArgumentParser(description='Enter a processed "Passages_[my file].csv" file to play a generation game.')
arg_parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder containing a Passages file with the following columns: Passage, Target, passage_no_target')

def_parser = WiktionaryParser()
def_parser.set_default_language('italian')

foldername = arg_parser.parse_args().data_folder[0]

passage_df = pd.read_csv(foldername + '/data/Passages.csv', encoding="utf-8")
word_df = pd.read_csv(foldername + '/data/words.csv', encoding="utf-8")

total_points = 0

#program loop
while True:
	randi = rd.randint(0, len(passage_df.index)-1)
	passage_row = passage_df.iloc[randi]
	passage_num = passage_row['PassageNum']
	word_passage_df = word_df >> mask(X.PassageNum == passage_num)
	# put this in format_raw
	word_passage_df >>= mutate(	
		#Sep = if_match_else(lead(X.CPOS), 'F', '', ' '),	
		Sep = if_else(lead(X.CPOS) == 'F', '', ' '),
		char_count = nchar(X.Word)		
	) >> mutate(
		char_count_cum = cumsum(X.char_count)
	) >> unite('Word_Sep', ['Word', 'Sep'], remove=False, sep="")	
	#print(word_passage_df	>> select(X.Sep, X.char_count_cum))
	print(word_passage_df >> unite('Word_CPOS', ['Word_Sep', 'CPOS'], remove=False, sep="")	>> select(X.Word_CPOS, X.char_count_cum))
	passage_no_target = concat_when(word_passage_df['Word'], word_passage_df['TargetFlag'], 0, '*___*')		
	passage_no_target = re.sub("(.{64})", r"\1 |\n", passage_no_target, 0, re.DOTALL)
	target = passage_row['Target']
	puts(colored.black('****************************************************************'))
	
	# Passage loop
	while True:
		break_all = False
		with indent(2, quote='|'):
			puts(colored.black(passage_no_target))
		puts(colored.black('\nInserisci le parole corrette per *___*. (e` -> è) \
			\n?parole per aiuto	\
			\n0 per uscire\n'))
		with indent(4, quote=' >'):
			user_words = prompt.query('>>> ')
		# replace o` with ò, replace e` with è, a` with à
		user_words = re.sub(r'o`', 'ò', user_words)
		user_words = re.sub(r'i`', 'ì', user_words)
		user_words = re.sub(r'e`', 'è', user_words)
		user_words = re.sub(r'a`', 'à', user_words)
		user_words = re.sub(r'u`', 'ù', user_words)

		if user_words == '0':
			break_all = True
			break;
		elif user_words[0] == "?":
			pprint_wiktionary(user_words[1:])
		else:
			points = 0
			user_words = user_words.split()
			target_word_df = word_passage_df >> mask(X.TargetFlag == 1)
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
				puts(colored.cyan('  _________\n /         \\\n |  O   O  |\n |    J    |\n \\  \\___/  /\n  \\_______/'))
			else:
				puts(colored.red(' Spiacente, solo ' + str(points) + " punti."))
				puts(colored.red('  _________\n /         \\\n |  /\\ /\\  |\n |    J    |\n \\   ___   /\n |  /   \\  |\n  \\_______/'))
			# break out of the Passage loop
			break
	if break_all:
		break
	passage = concat_when(word_passage_df['Word'], word_passage_df['TargetFlag'], 0, '*{word}*')		
	passage = re.sub("(.{64})", r"\1 |\n", passage, 0, re.DOTALL)
	with indent(2, quote='|'):
		puts(colored.black(passage))
	puts("\n")
	puts(colored.magenta('(Ha '+ str(total_points) + ' punti.)'))