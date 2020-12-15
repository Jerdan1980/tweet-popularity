from models import *
import csv
import random
import nltk
import sparse #sparse matrices

# import custom file
from models import *

# create lists to hold data
lexicon = []
tweets = []
retweets = []
likes = []
data = []

# https://realpython.com/python-csv/
# import csv
with open('corona.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    linecount = 0
    for row in csv_reader:
        # remove blank lines
        if len(row) < 3:
            continue

        # save it
        data.append(row)

        # add tweet's dictionary to lexicon
        # it doesnt matter what order they get added so its fine here
        lexicon += list(word_tokenize(row[0]))

        linecount += 1
    print(f'Imported {linecount} tweets')

# remove dupes from lexicon
lemmatizer = WordNetLemmatizer()
lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
print(f'Lexicon has {len(lexicon)} words')

# shuffle data now so i dont have to do it later
random.shuffle(data)

# split the data up
for d in data:
    tweets.append(d[0])
    retweets.append(d[1])
    likes.append(d[2])
# delete data to make space
data = None
print(f'Shuffled {len(tweets)} tweets')

# sentdex dictionary
# sentdexdict = modifiedsentdex(lexicon)
# print(f'Sentdex dictionary has {len(sentdexdict)} words')

# n-gram dictionary
featureset = generateNGram(tweets, lexicon)
print("Finished n-gram dictionary")

# simple dictionary
#featureset = createMatrix(tweets, lexicon)
#print("Finished simple dictionary model")

# convert to numpy
featureset = np.array(featureset)
# divide data, 90% goes to training
train_size = int(.9 * len(featureset))
endMarker = len(lexicon) - 1
# convert them all to double arrays so that it can do the thing.
# otherwise they will throw a fit due to loss of information.
trainMatrix = np.array(featureset[:][:train_size], np.double)
trainRetweets = np.array(retweets[:train_size], np.double)
trainLikes = np.array(likes[:train_size], np.double)
testMatrix = np.array(featureset[:][train_size:], np.double)
testRetweets = np.array(retweets[train_size:], np.double)
testLikes = np.array(likes[train_size:], np.double)
print("Finished partitioning data")

(rtWeights, likeWeights) = solve(trainMatrix, trainRetweets, trainLikes)

# calculate the errors
(rtErr, likeErr) = loss(testMatrix, testRetweets, testLikes, rtWeights, likeWeights)
print(f'Testing on {len(testRetweets)} tweets:')
print(f'\tRetweet error: {rtErr}')
print(f'\tLike error: {likeErr}')