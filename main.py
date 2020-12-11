import csv
import random

#used to create a dictionary
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle #is needed by nltk... somehow
from collections import Counter
from nltk.stem import WordNetLemmatizer

#import custom file
from models import *

#create lists to hold data
lexicon = []
tweets = []
retweets = []
likes = []
data = []

#https://realpython.com/python-csv/
#import csv
with open('tweets.csv', encoding='utf-8') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	linecount = 0
	for row in csv_reader:
		#remove blank lines
		if len(row) < 3:
			continue

		#save it
		data.append(row)

		#add tweet's dictionary to lexicon
		#it doesnt matter what order they get added so its fine here
		lexicon += list(word_tokenize(row[0]))
		
		linecount += 1
	print(f'Imported {linecount} tweets')

# remove dupes from lexicon
lemmatizer = WordNetLemmatizer()
lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
print(f'Lexicon has {len(lexicon)} words')

#shuffle data now so i dont have to do it later
random.shuffle(data)

#split the data up
for d in data:
	tweets.append(d[0])
	retweets.append(d[1])
	likes.append(d[2])
#delete data to make space
data = None
print(f'Shuffled {len(tweets)} tweets')

#sentdex dictionary
sentdexdict = modifiedsentdex(lexicon)
print(f'Sentdex dictionary has {len(sentdexdict)} words')

#simple dictionary
featureset = createMatrix(tweets, lexicon)
print("Finished simple dictionary model")

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

(rtWeights, likeWeights) = solve(trainMatrix, trainRetweets, trainLikes)

#test one tweet out
print(f'Estimated retweets: {testMatrix[:][1] @ rtWeights}')
print(f'Actual Retweets: {testRetweets[1]}')
print(f'Estimated retweets: {testMatrix[:][1] @ likeWeights}')
print(f'Actual Retweets: {testLikes[1]}')