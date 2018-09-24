#coding: utf-8
import os, sys, math, re

import argparse
import pandas as pd
import numpy as np
from dfply import *
from clint.textui import puts, indent, colored, prompt, validators
from wiktionaryparser import WiktionaryParser



@make_symbolic
def nchar(words):
	return [len(x) for x in list(words)]

@make_symbolic
def np_where(bools, val_if_true, val_if_false):
	return list(np.where(bools, val_if_true, val_if_false))

@make_symbolic
def re_search_any(series, pattern):
	out = []
	for s in series:
		n = re.search(pattern, s)
		if n: 
			out.append(True)
		else:
			out.append(False)
	return out


@make_symbolic
def pprint_passage(words_df, col_to_print, char_width = 80):
	# the unite function has a stray print, supress it (hopefully this can be changed)
	num_tokens = max(words_df.token_num)
	sys.stdout = open(os.devnull, 'w')
	words_df['To_Print'] = [re.sub(r'\s', ' ', s) for s in list(words_df[col_to_print])]
	words_df >>= (
		mutate(	
			sep = np_where((lead(X.cpos) == 'F') & (lead(X.To_Print) != '(') | \
				re_search_any(X.To_Print, r".+?'$") \
				, '', ' '),			
		) >> unite('word_sep', ['To_Print', 'sep'], remove=False, sep="")	>> 
		mutate(
			char_count = nchar(X.word_sep)	
		) 
	)
	sys.stdout = sys.__stdout__
	# to get lines we need to iterate through the dataframe, finding groups of characters <= char_width
	char_count = 0
	line_num = 1
	words_df['char_start'] = 0
	words_df['char_stop'] = 0
	words_df['line_num'] = line_num
	words_df['word_sep_nl'] = words_df['word_sep']
	for i, row in words_df.iterrows():
		#print(i)
		# start a new line?
		if char_count + words_df.loc[i,'char_count'] > char_width:
			char_count = 0
			line_num += 1

			# update the previous line
			previous_text = words_df.loc[i-1, 'word_sep_nl']
			buffer_text = (" " * math.floor(char_width - words_df.loc[i-1,'char_stop'] )) + "|\n"
			words_df.loc[i-1, 'word_sep_nl'] = previous_text + buffer_text

		words_df.loc[i, 'char_start'] = char_count
		char_count += words_df.loc[i,'char_count']
		words_df.loc[i,'line_num'] = line_num
		words_df.loc[i,'char_stop'] = char_count

	# update the final line
	#i = len(words_df)
	previous_text = words_df.loc[i-1, 'word_sep_nl']
	buffer_text = (" " * math.floor(char_width - words_df.loc[i-1,'char_stop'] )) + "|\n"
	words_df.loc[i-1, 'word_sep_nl'] = previous_text + buffer_text
	
	final_words = words_df.word_sep_nl.str.cat(sep='')
	puts(colored.black(' ' + ('_' * (char_width+1))))
	with indent(2, quote='|'):
		puts(colored.black(final_words))
	puts(colored.black(' ' + ('_' * (char_width+1))))
	
	# remove fields created here
	words_df >>= drop(['To_Print', 'sep', 'char_count', 'char_start','char_stop', 'line_num', 'word_sep', 'word_sep_nl'])
	return final_words

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
						if k == 0 and definition['text'][k] != word:
							print(definition['text'][k] + ":")
						elif k > 0:
							print(str(k) + '. ' + definition['text'][k])	
				if definition['relatedWords'] and len(definition['relatedWords']) > 0:
					for m in range(len(definition['relatedWords'])):
						related = definition['relatedWords'][m]
						if related['relationshipType'] and len(related['words']) > 0:
							print(' --- ' + related['relationshipType'] + ' ---')
							print(related['words'])	
				if definition['examples'] and len(definition['examples']) > 0:
					print(' --- Examples ---')
					for m in range(len(definition['examples'])):
						print(definition['examples'][m])	
		print('\n\n\n')					
	#except:
	#	print("couldn't access Wiktionary")

arg_parser = argparse.ArgumentParser(description='Enter a processed "Passages_[my file].csv" file to play a generation game.')
arg_parser.add_argument('data_folder', metavar='N', type=str, nargs='+',
                   help='a folder containing a Passages file with the following columns: Passage, Target, passage_no_target')

def_parser = WiktionaryParser()
def_parser.set_default_language('italian')

foldername = arg_parser.parse_args().data_folder[0]

#passage_df = pd.read_csv(foldername + '/data/Passages.csv', encoding="utf-8")
words_df = pd.read_csv(foldername + '/data/words.csv', encoding="utf-8")

total_points = 0

#program loop
while True:
	passage_num = np.random.randint(min(words_df.passage_num) + 1, max(words_df.passage_num)-1)
	words_passage_df = words_df >> mask(X.passage_num == passage_num)
	
	words_passage_df >>= mutate(
		words_no_target = np_where(X.target_flag == 1, '*___*', X.word),
		words_bold_target = np_where(X.target_flag == 1, '*' + X.word + '*', X.word)
	)
	# Passage loop
	while True:
		break_all = False
		# replace the target words with *___*
		passage_no_target = pprint_passage(words_passage_df, 'words_no_target')		
		
		puts(colored.black('''
			Inserisci le parole corrette per *___*. (e` → è, e^ → é)
			?parole per aiuto
			trad paroli... (tradurre paroli) o trad (tradurre tutti)
			0 per uscire
			'''))
		with indent(4, quote=' >'):
			user_words = prompt.query('>>> ')
		# replace o` with ò, replace e` with è, a` with à
		user_words = re.sub(r'o`', 'ò', user_words)
		user_words = re.sub(r'i`', 'ì', user_words)
		user_words = re.sub(r'e`', 'è', user_words)
		user_words = re.sub(r'a`', 'à', user_words)
		user_words = re.sub(r'u`', 'ù', user_words)
		user_words = re.sub(r'e\^', 'é', user_words)
		
		if user_words == '0':
			break_all = True
			break;
		elif user_words == 'debug':
			print(words_passage_df)
		elif user_words[0] == "?":
			pprint_wiktionary(user_words[1:])
		elif user_words[0:4] == 'trad':
			if len(user_words) > 5:
				print(translator.translate(user_words[5:]))
			else:
				print(translator.translate(passage_no_target))
		else:
			points = 0
			user_words = user_words.split()
			target_words_df = words_passage_df >> mask(X.target_flag == 1)
			for i in range(len(user_words)):
				user_word = user_words[i]
				target_word = target_words_df['word'].iloc[i]
				target_lemma = target_words_df['lemma'].iloc[i]
				target_cpos = target_words_df['cpos'].iloc[i]
				target_lemma_cpos = target_lemma + "|" + target_cpos
				# get all instances of the words df that match the user's word, used to find lemma-cpos
				lemma_cposs = list(words_df >> mask(X.word == user_word) >> 
					distinct(X.word, X.lemma, X.cpos) >>
					mutate(lemma_cpos = X.lemma + "|" + X.cpos) >>
					pull('lemma_cpos')
				)
				if user_word == target_word:
					points += 3
				elif target_lemma_cpos in lemma_cposs:
					points += 1
			total_points += points
			if points > 2:
				puts(colored.cyan(' Ben fatto, ' + str(points) + " punti."))
				puts(colored.cyan('  _________\n /  -   -  \\\n |  O   O  |\n |    J    |\n \\  \\___/  /\n  \\_______/'))
			else:
				puts(colored.red(' Spiacente, solo ' + str(points) + " punti."))
				puts(colored.red('  _________\n /         \\\n | /_\\ /_\\ |\n |    J    |\n \\   ___   /\n  \\ /   \\ /\n   \\_____/'))
			# break out of the Passage loop
			break
	if break_all:
		break
	
	pprint_passage(words_passage_df, 'words_bold_target')
	puts(colored.magenta('(Ha '+ str(total_points) + ' punti.)'))