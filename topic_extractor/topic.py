#!/usr/bin/python3

qqn="MLP_officiel"
f = open(qqn+"_tweets.txt","r+")
documents= f.readlines()
f.close()


# renvoies les mots cles
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#import nltk.twitter  as twi
from string import punctuation
from lda import lda,utils
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer


punct=[p for p in punctuation]
stoplist = set(stopwords.words("french"))
stoplist.update(punct)
## en utilisant sklearn
count_vect = CountVectorizer()
count_vect.stop_words=stoplist
X_train_counts = count_vect.fit_transform(documents)

vocab = [a for a,b in sorted(list(count_vect.vocabulary_.items()),key=lambda
    tup:tup[1])]

model = lda.LDA(n_topics=5, n_iter=50)
model.fit(X_train_counts)
topic_word = model.topic_word_
n_top_words=5



