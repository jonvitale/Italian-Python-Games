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
def pprint_passage(words_df, col_to_print, char_width = 12):
	# the unite function has a stray print, supress it (hopefully this can be changed)
	num_tokens = max(words_df.TokenNum)
	sys.stdout = open(os.devnull, 'w')
	words_df >>= (
		mutate(	
			Sep = np_where((lead(X.CPOS) == 'F') & (lead(X[col_to_print]) != '(') | \
				re_search_any(X[col_to_print], r".+?'$") \
				, '', ' '),			
		) >> unite('Word_Sep', [col_to_print, 'Sep'], remove=False, sep="")	>> 
		mutate(
			CharCount = nchar(X.Word_Sep)	
		) >> mutate(
			CharStart = cumsum(lag(X.CharCount))
		)
	)
	words_df['CharStart'].fillna(0, inplace=True)	
	words_df >>= mutate(CharStop = X.CharStart + X.CharCount)
	words_df['LineNum'] = [math.floor(x / char_width) + 1 for x in words_df.CharStart]
	
	# add a newline after every last word
	words_df >>= mutate(Overflow = char_width*X.LineNum - X.CharStop)
	 #>> mutate(
	#	NL = np_where((X.Overflow <= 0) & (X.TokenNum != num_tokens), '|\n', '')
	#) >> unite('Word_Sep_NL', ['Word_Sep', 'NL'], remove=False, sep="")
	
	# what should we extend every end of line to?
	words_df['Word_Sep_NL'] = words_df['Word_Sep']
	max_overflow = max(-1 * words_df.Overflow)
	sys.stdout = sys.__stdout__
	for i in range(len(words_df)):		
		if words_df.Overflow.iloc[i] <= 0:
			words_df.Word_Sep_NL.iloc[i] += (" " * math.floor(max_overflow + words_df.Overflow.iloc[i])) + " |\n"
			print(str(max_overflow) + ' + ' +str(words_df.Overflow.iloc[i]) + ' = ' + str(words_df.CharStop.iloc[i]) + " - " + str(words_df.Word_Sep_NL.iloc[i]))
			

	print(words_df)
	puts(colored.black(' __________________________________________________________________'))
	with indent(2, quote='|'):
		puts(colored.black(words_df.Word_Sep_NL.str.cat(sep='')))
	puts(colored.black(' __________________________________________________________________'))
	
	# remove fields created here
	words_df >>= drop(['Sep', 'CharCount', 'CharStart','CharStop', 'LineNum', 'Word_Sep', 'Word_Sep_NL'])
	

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
	passage_num = np.random.randint(min(words_df.PassageNum) + 1, max(words_df.PassageNum)-1)
	words_passage_df = words_df >> mask(X.PassageNum == passage_num)
	
	#print(words_passage_df >> select(X.Word, X.Word_Sep, X.CharCount, X.CharStart))

	#print(words_passage_df >> unite('Word_CPOS', ['Word_Sep', 'CPOS'], remove=False, sep="")	>> select(X.Word_CPOS, X.char_count_cum))
	#passage_no_target = concat_when(words_passage_df['Word_Sep'], words_passage_df['TargetFlag'], 0, '*___* ', sep="")		
	#passage_no_target = re.sub("(.{64})", r"\1 |\n", passage_no_target, 0, re.DOTALL)
	#target = passage_row['Target']
	
	words_passage_df >>= mutate(
			Words_no_target = np_where(X.TargetFlag == 1, '*___*', X.Word))

	# Passage loop
	while True:
		break_all = False
		# replace the target words with *___*
		pprint_passage(words_passage_df, 'Words_no_target')
		
		
		puts(colored.black('\nInserisci le parole corrette per *___*. (e` → è, e^ → é) \
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
		user_words = re.sub(r'e\^', 'é', user_words)
		
		if user_words == '0':
			break_all = True
			break;
		elif user_words[0] == "?":
			pprint_wiktionary(user_words[1:])
		else:
			points = 0
			user_words = user_words.split()
			target_words_df = words_passage_df >> mask(X.TargetFlag == 1)
			for i in range(len(user_words)):
				user_word = user_words[i]
				target_word = target_words_df['Word'].iloc[i]
				target_lemma = target_words_df['Lemma'].iloc[i]
				target_cpos = target_words_df['CPOS'].iloc[i]
				target_lemma_cpos = target_lemma + "|" + target_cpos
				# get all instances of the words df that match the user's word, used to find lemma-cpos
				lemma_cposs = list(words_df >> mask(X.Word == user_word) >> 
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
				puts(colored.cyan('  _________\n /  -   -  \\\n |  O   O  |\n |    J    |\n \\  \\___/  /\n  \\_______/'))
			else:
				puts(colored.red(' Spiacente, solo ' + str(points) + " punti."))
				puts(colored.red('  _________\n /         \\\n | /_\\ /_\\ |\n |    J    |\n \\   ___   /\n  \\ /   \\ /\n   \\_____/'))
			# break out of the Passage loop
			break
	if break_all:
		break
	passage = concat_when(words_passage_df['Word'], words_passage_df['TargetFlag'], 0, '*{word}*')		
	passage = re.sub("(.{64})", r"\1 |\n", passage, 0, re.DOTALL)
	with indent(2, quote='|'):
		puts(colored.black(passage))
	puts("\n")
	puts(colored.magenta('(Ha '+ str(total_points) + ' punti.)'))