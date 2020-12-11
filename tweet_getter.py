#!/usr/bin/python
import tweepy
import csv
import time
import sys
import io
import re
import json

#import json file to support multiple people without revealing secrets
with open("tokens.json") as f:
	tokens = json.load(f)

#create API Handler
auth = tweepy.auth.OAuthHandler(tokens["API Key"], tokens["API Secret"])
auth.set_access_token(tokens["Access Token"], tokens["Access Secret"])
api = tweepy.API(auth)

#open csv
csvFile = open('tweets.csv', 'a', encoding="utf-8")

#Use csv writer
csvWriter = csv.writer(csvFile)

#twitter will stop you after two searches, so manually change this accordingly.
queries = ["coronavirus", "snowboarding", "the game awards"]

#cycle through queries
for query in queries:
	print(query)
	#get tweets in search parameters
	#limit to 2k since it throws an error a little after 2k
	for tweet in tweepy.Cursor(api.search,
														q = "coronavirus",
														since = "2020-12-5",
														until = "2020-12-11",
														tweet_mode="extended",
														lang = "en").items(2000):

		# Write the tweets with no commas, the likes and retweet count into the csv
		try:
				fav_count = (tweet.retweeted_status.favorite_count)
		except:
				fav_count = (tweet.favorite_count)

		#grab the full tweet
		try:
			msg = tweet.retweeted_status.full_text
		except AttributeError:
			msg = tweet.full_text

		#remove commas and newlines
		msg = msg.replace(",", "")
		msg = msg.replace("\n", " ")

		#remove the "RT @whatever: "
		msg = re.sub("^RT\s@\w+: ", "", msg)

		csvWriter.writerow([msg, tweet.retweet_count, fav_count])

#print statement so you know it finished.
print("Done!")