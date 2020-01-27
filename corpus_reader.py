
import os
from mwe import Mwe
import pickle

class Corpus_reader:
	"""
		This class is used to read the corpus (using class Corpus) and convert the labels to IOBo .
		o is the tag for tokens in between the components of an MWE which do not belong to the MWE.
		o was referred to as G in the paper.
	"""

	def __init__(self, path):

		mweCorpus = Mwe(path)
		self.train_sents = mweCorpus.sent_extractor(mweCorpus.train_collection)
		#self.dev_sents = mweCorpus.sent_extractor(mweCorpus.dev_collection)
		self.test_sents = mweCorpus.sent_extractor(mweCorpus.test_collection)
	

	def read(self, sents):
		seqs = []
		for i in sents:
			seqs_i = []
			for j in i:
				j_list = j.split('\t')
				seqs_i.append((j_list[1], j_list[2], j_list[3], int(j_list[5]), j_list[6], j_list[7]))
			seqs.append(seqs_i)
		
		return seqs
	
	"""
	def read(self, sents):
		seqs = []
		for s in range(0, len(sents)):
			seqs_i = []
			active_mwe = 0
			for t in range(len(sents[s].tokens)):
				if len(sents[s].tokens[t].parentMWEs) > 0:
					tag = ''
					mwe = sents[s].tokens[t].parentMWEs[0]
					if sents[s].tokens[t] == mwe.tokens[0]: # Check if the token is the first components of MWE
						tag = tag + 'B_'
						if not mwe.isSingleWordExp:
							active_mwe = 1
					else:
						tag = tag + 'I_'
						if sents[s].tokens[t] == mwe.tokens[-1]: # Check if the token is the first components of MWE
							active_mwe = 0
					tag = tag + mwe.type
					for mwe in sents[s].tokens[t].parentMWEs[1:]:
						if sents[s].tokens[t] == mwe.tokens[0]:
							tag = tag + ';B_'
						else:
							tag = tag + ';I_'
						tag = tag + mwe.type

				elif active_mwe:
					tag = 'o_' + mwe.type
				else:
					tag = 'O'

				seqs_i.append((sents[s].tokens[t].text, sents[s].tokens[t].lemma,
                                sents[s].tokens[t].posTag, sents[s].tokens[t].dependencyParent,
                                sents[s].tokens[t].dependencyLabel, tag))
			seqs.append(seqs_i)
		return (seqs)


### A sample code to test Corpus_reader
'''
c = Corpus_reader("./ES/")
print("train sents",len(c.train_sents))
print("dev sents",len(c.dev_sents))
print("test sents",len(c.test_sents))
s = c.read(c.test_sents)
for i in s[7:10]:
	for j in i:
		print(j)
	print()
'''
"""