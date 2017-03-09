#!/usr/bin/python3

import os
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

# concatène les données dans une table [texte, sentiment]
def getData():
    positive_files_path = './txt_sentoken/pos/'
    negative_files_path = './txt_sentoken/neg/'

    positive_files = [positive_files_path + pos_file for pos_file in os.listdir(positive_files_path)]
    negative_files = [negative_files_path + pos_file for pos_file in os.listdir(negative_files_path)]

    positive_texts = []
    negative_texts = []

    for f in positive_files:
        with open(f, 'r') as posfile:
            positive_texts.append(posfile.read())

    for f in negative_files:
        with open(f, 'r') as negfile:
            negative_texts.append(negfile.read())

    pos_dataframe = pd.DataFrame(positive_texts)
    neg_dataframe = pd.DataFrame(negative_texts)

    pos_dataframe['sentiment'] = 1
    neg_dataframe['sentiment'] = 0

    yelp_df          = pd.read_csv("yelp_labelled.txt", delimiter='\t')
    imdb_df          = pd.read_csv("imdb_labelled.txt", delimiter='\t')
    amazon_cells_df  = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t')
    imdb_training_df = pd.read_csv("training.txt", delimiter='\t', names=[0, 1])

    # Gotta swap 'em all
    imdb_training_df[0], imdb_training_df[1] = imdb_training_df[1], imdb_training_df[0]

    text_sentiment_columns = ["Text", "Sentiment"]

    # assigne des noms de colonnes significatifs aux données classifiées ...

    pos_dataframe.columns = text_sentiment_columns
    neg_dataframe.columns = text_sentiment_columns

    # assigne des noms de colonnes significatifs aux données NON classifiées ...
    yelp_df.columns          = text_sentiment_columns
    imdb_df.columns          = text_sentiment_columns
    amazon_cells_df.columns  = text_sentiment_columns
    imdb_training_df.columns = text_sentiment_columns


    # on rassemble les sets de données
    total_df = [pos_dataframe, neg_dataframe, imdb_training_df, yelp_df, imdb_df, amazon_cells_df]

    return pd.concat(total_df)

# divise les données en set d'entrainement selon le pourcentage (e.g splitWhere=80 => retourne [[80% des données], [20% des données]])
def data_split(dataframe, splitWhere):
    splitWhere = int(splitWhere * len(dataframe))
    df = shuffle(dataframe)

    # Check splitting
    train = [(dict(text=Text), Sentiment) for index, (Text, Sentiment) in df.iterrows()][:splitWhere]
    test = [(dict(text=Text), Sentiment) for index, (Text, Sentiment) in df.iterrows()][splitWhere:]
    # print (" X_train length: {0}\n y_train length: {1}\n X_test_length: {2}\n y_test_length: {3}\n".format(len(X_train), len(y_train), len(X_test), len(y_test)))

    return (train, test)

# Chaque mot correspond à une règle d'implication du classifieur, on affiche en ordonnées le poids de cette règle pour la classification.
def visualize_data(classifier, feature_names, n_top_features=25):
    coef = classifier.coef_.ravel()
    pos_coef = np.argsort(coef)[-n_top_features:]
    neg_coef = np.argsort(coef)[:n_top_features]
    interesting_coefs = np.hstack([neg_coef, pos_coef])

    # prépare le graphe avec les données.
    plt.figure(figsize=(20, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[interesting_coefs]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefs], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefs], rotation=60, ha='right');


trainFrame, testFrame = data_split(getData(), 90.0/100.0)

print(testFrame)

# classifieur de bayes
classifier = nltk.NaiveBayesClassifier.train(trainFrame)
print ('précision :', nltk.classify.util.accuracy(classifier, testFrame))

# enregistre le classifieur entrainté dans un fichier
f = open('naivebayes.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

# load classifier model
# classifier_f = open("naivebayes.pickle", "rb")
# classifier2 = pickle.load(classifier_f)
# classifier_f.close()

print(classifier.classify_many((
    dict(text="Always knows what I want, not guy crazy, hates Harry Potter"),
    dict(text="I never come again."),
    dict(text="we need to get joe out of here"),
    dict(text="Brokeback Mountain is fucking horrible.."),
    dict(text="is broke and in a dead end job")
)))
