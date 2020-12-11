import csv
import random

#used to create a dictionary
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle #is needed by nltk... somehow
from collections import Counter
from nltk.stem import WordNetLemmatizer

#cupy for CUDA SPEEDS
import cupy as cp

#create lists to hold data
lexicon = []
tweets = []
retweets = []
likes = []

#https://realpython.com/python-csv/
with open('tweets.csv', encoding='utf-8') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	linecount = 0
	for row in csv_reader:
		#remove blank lines
		if len(row) < 3:
			continue

		#text, retweets, likes
		tweets.append(row[0])
		retweets.append(row[1])
		likes.append(row[2])

		#add tweet's dictionary to lexicon
		lexicon += list(word_tokenize(row[0]))
		
		linecount += 1
	print(f'Processed {linecount} lines')

#https://pythonprogramming.net/preprocessing-tensorflow-deep-learning-tutorial/
lemmatizer = WordNetLemmatizer()
# remove dupes
lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
print(f'Lexicon has {len(lexicon)} words')

#simple dictionary from in-class
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
print("Finished simple dictionary")

#sentdex's version
w_counts = Counter(lexicon)
trunc_list = []
for w in w_counts:
	#change this to be percentage wise?
	#theres a world of difference between 25 and 0
	#not much is going on above a thousand
	if 1000 > w_counts[w] > 25:
		trunc_list.append(w)
print("Finished sentdex dictionary")

#shuffle
#random.shuffle(featureset)
#cant figure out slice notation so I am not gonna try shuffling yet
#convert to numpy
featureset = np.array(featureset)
#divide data, 90% goes to training
train_size = int(.9*len(featureset))
endMarker = len(lexicon)-1
#convert them all to double arrays so that it can do the thing.
#otherwise they will throw a fit due to loss of information.
trainMatrix = np.array(featureset[:][:train_size], np.double)
trainRetweets = np.array(retweets[:train_size], np.double)
trainLikes = np.array(likes[:train_size], np.double)
testMatrix = np.array(featureset[:][train_size:], np.double)
testRetweets = np.array(retweets[train_size:], np.double)
testLikes = np.array(likes[train_size:], np.double)
print("Finished partitioning data")


rtWeights = np.linalg.lstsq(trainMatrix, trainRetweets, rcond=None)[0] #silence the warning
likeWeights = np.linalg.lstsq(trainMatrix, trainLikes, rcond=None)[0] #silence the warning

#test one tweet out
print(f'Estimated retweets: {testMatrix[:][1] @ rtWeights}')
print(f'Actual Retweets: {testRetweets[1]}')
print(f'Estimated retweets: {testMatrix[:][1] @ likeWeights}')
print(f'Actual Retweets: {testLikes[1]}')

print(trainMatrix.nbytes)