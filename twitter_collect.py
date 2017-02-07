#!/usr/bin/python3

import csv
import tweepy
from tweepy import OAuthHandler

consumer_key = 'VA43ZIj7GjxIgH55YY3fwcXrM'
consumer_secret = 'qOOhdAWa8yI0ENyjBFTfmy35DziTNqr39s5vfDRI2rXZuLudpw'
access_token = '827434915516055552-jAvdBPGYHFp8vW5V7a4qd0cscf8shRZ'
access_secret = 'P9ojr2aMj85DSpM1PEKtqloR6xBXCD8WCukeSGfIFqTbj'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

politichien = "dupontaignan"

newTweets = api.user_timeline(screen_name=politichien, count=200)

with open("%s_tweets.csv" % politichien, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "created_at", "text"])
    writer.writerows(newTweets)
