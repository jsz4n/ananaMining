#!/usr/bin/python3

import tweepy
from tweepy import OAuthHandler

consumer_key = 'VA43ZIj7GjxIgH55YY3fwcXrM'
consumer_secret = 'qOOhdAWa8yI0ENyjBFTfmy35DziTNqr39s5vfDRI2rXZuLudpw'
access_token = '827434915516055552-jAvdBPGYHFp8vW5V7a4qd0cscf8shRZ'
access_secret = 'P9ojr2aMj85DSpM1PEKtqloR6xBXCD8WCukeSGfIFqTbj'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text) 
