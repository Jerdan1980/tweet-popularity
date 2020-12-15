import numpy as np
import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
import numpy as np
import pickle  # is needed by nltk... somehow
from collections import Counter
from nltk.stem import WordNetLemmatizer
from numba import jit  # for SPEEDUP


def generateNGram(lexicon):
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, lexicon)
    model = MLE(n)
    model.fit(train_data, padded_sents)
    featureset = []
    for i in lexicon:
        featureset.append(model.counts[i])

    return featureset


# the simple 1:1 dictionary we learned in class
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
        # create augmented matrix of words and the retweets, since that seems to work better
        featureset.append(list(features))
    return featureset


# a modified dectionary based off of sentdex's deeplearning tutorial
# https://pythonprogramming.net/preprocessing-tensorflow-deep-learning-tutorial/
def modifiedsentdex(lexicon):
    # creates a map of word to its count
    w_counts = Counter(lexicon)

    # make an array of just the counts
    counts = []
    for w in w_counts:
        counts.append(w_counts[w])

    # find the upper and lower quartiles
    lowerQuartile = np.percentile(counts, 25)
    upperQuartile = np.percentile(counts, 90)
    print(f'\tCutoffs: {lowerQuartile}, {upperQuartile}')
    print(f'\tMax: {np.amax(counts)}')

    trunc_list = []
    for w in w_counts:
        # remove anything not in the middle 50%
        if upperQuartile >= w_counts[w] >= lowerQuartile:
            trunc_list.append(w)
    return trunc_list


# @jit(nopython=True) #for SPEEDUP
def solve(M, RTs, Likes):
    # lstsq returns 4 outputs: x, resid, rank, and "singular values of a".
    # we only want x, the first output.
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    # rcond both silences the futurewarning and sets cutoff values of the matrix
    rtWeights = np.linalg.lstsq(M, RTs, rcond=None)[0]
    likeWeights = np.linalg.lstsq(M, Likes, rcond=None)[0]
    return (rtWeights, likeWeights)


# @jit
def loss(M, RTs, Likes, rtWeights, likeWeights):
    # set initial counts to zero
    rtErr = 0
    likeErr = 0

    # summ error squared
    for i in range(0, len(RTs)):
        rtErr += np.abs((M[:][i] @ rtWeights) - RTs[i])
        likeErr += np.abs((M[:][i] @ likeWeights) - Likes[i])

    # divide by n
    rtErr = rtErr / len(RTs)
    likeErr = likeErr / len(Likes)

    return (rtErr, likeErr)
