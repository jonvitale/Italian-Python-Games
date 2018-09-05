import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser(description='Enter raw KWIC data from PAISA database (from corpus search).')
parser.add_argument('kwic_file', metavar='N', type=str, nargs='+',
                   help='a kwic data file from PAISA')

filename = parser.parse_args().kwic_file[0]


with open(filename, encoding="utf-8") as in_file:
	filename = re.search(r'(.*)?(\.kwic?)\.txt', filename).group(1)
	print(filename)
	sentences = in_file.readlines()
	# encode all sentences
	# sentences = [s.encode('utf-8') for s in sentences]
	# get sentence w/o target and target
	sentences_no_target = []
	targets = []
	for s in sentences:
		matches = re.search(r'(.*)?<(.*?)>(.*)', s)
		targets.append(matches.group(2))
		sentences_no_target.append(matches.group(1) + "?___?  ?___?" + matches.group(3))
	df = pd.DataFrame()
	df['Sentence'] = pd.Series(sentences) 
	df['Target'] = pd.Series(targets) 
	df['Sentence_no_target'] = pd.Series(sentences_no_target) 
	df.to_csv('sentences_'+filename+".csv")