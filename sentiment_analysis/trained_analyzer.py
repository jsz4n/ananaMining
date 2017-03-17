#!/usr/bin/python3

import sys
import pickle
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random
import pprint
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


# load classifier model
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


if len(sys.argv) > 1:
    sentences=[]
    for sentence in sys.argv[1:]:
        sentences.append(dict(text=sentence))

    sentiments = classifier.classify_many(sentences)

    for sentence, sentiment in zip(sentences, sentiments):
        print(sentence['text'], " => ", 'positive' if sentiment else 'negative')

else:
    print(classifier.classify_many((
        dict(text="always knows what I want, not guy crazy, hates Harry Potter"),
        dict(text="I never come again."),
        dict(text="we need to get joe out of here"),
        dict(text="Brokeback Mountain is fucking horrible.."),
        dict(text="is broke and in a dead end job")
    )))
