import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle #is needed by nltk... somehow
from collections import Counter
from nltk.stem import WordNetLemmatizer
from numba import jit #for SPEEDUP

#the simple 1:1 dictionary we learned in class
def createMatrix(tweets, lexicon):
	lemmatizer = WordNetLemmatizer()
	featureset = []
	for t in tweets:
		words = word_tokenize(t)
		words = [lemmatizer.lemmatize(i) for i in words]
		features = np.zeros(len(lexicon))
		for w in words:
			if w in lexicon:
				features[lexicon.index(w)] += 1
		#create augmented matrix of words and the retweets, since that seems to work better
		featureset.append(list(features))
	return featureset

#a modified dectionary based off of sentdex's deeplearning tutorial
#https://pythonprogramming.net/preprocessing-tensorflow-deep-learning-tutorial/
def modifiedsentdex(lexicon):
	#creates a map of word to its count
	w_counts = Counter(lexicon)

	#make an array of just the counts
	counts = []
	for w in w_counts:
		counts.append(w_counts[w])
	
	#find the upper and lower quartiles
	lowerQuartile = np.percentile(counts, 25)
	upperQuartile = np.percentile(counts, 75)
	print(f'Cutoffs: {lowerQuartile}, {upperQuartile}')

	trunc_list = []
	for w in w_counts:
		#remove anything not in the middle 50%
		if 10 > w_counts[w] > 5:
			trunc_list.append(w)
	return trunc_list

#@jit(nopython=True) #for SPEEDUP
def solve(M, RTs, Likes):
	#lstsq returns 4 outputs: x, resid, rank, and "singular values of a".
	#we only want x, the first output.
	#https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
	#rcond both silences the futurewarning and sets cutoff values of the matrix
	rtWeights = np.linalg.lstsq(M, RTs, rcond=None)[0]
	likeWeights = np.linalg.lstsq(M, Likes, rcond=None)[0]
	return (rtWeights, likeWeights)