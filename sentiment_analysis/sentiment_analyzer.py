#!/usr/bin/python3

import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def debug(*dfs):
    for count, df in enumerate(dfs):
        print("DF No. {0}".format(count + 1))
        print(df.head())

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

    X_train = [text for text in df.iloc[:splitWhere,:1]['Text']]
    y_train = [sentiment for sentiment in df.iloc[:splitWhere, :2]['Sentiment']]
    X_test = [text for text in df.iloc[splitWhere: , :1]['Text']]
    y_test = [sentiment for sentiment in df.iloc[splitWhere: , :2]['Sentiment']]

    # Check splitting
    # print (" X_train length: {0}\n y_train length: {1}\n X_test_length: {2}\n y_test_length: {3}\n".format(len(X_train), len(y_train), len(X_test), len(y_test)))

    return ((X_train, y_train), (X_test, y_test))

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

def browse_facebook():
    browser = webdriver.Firefox()

    # Email and password elements
    username_xpath = "//input[@id='email']"
    password_xpath = "//input[@id='pass']"
    login_button_xpath = '//*[@id="loginbutton"]'

    browser.get('https://www.facebook.com')

    username_element = browser.find_element_by_xpath(username_xpath)
    password_element = browser.find_element_by_xpath(password_xpath)
    loginbutton = browser.find_element_by_xpath(login_button_xpath)

    # Writing the username and password
    username_element.send_keys('thrashpoubelle@gmail.com')
    password_element.send_keys('Zeuros00;')

    # Logging in
    loginbutton.click()

    # Maximizing the browser [optional]
    browser.maximize_window()

    browser.get('https://www.facebook.com/DonaldTrump/')

    # Getting the body for scrolling
    body = browser.find_element_by_tag_name('body')

    numberofscrolls = 100

    # Ignoring the pop ups
    for i in range(10):
        body.send_keys(Keys.ESCAPE)
        browser.implicitly_wait(1)

    # Scrolling
    for i in range(numberofscrolls):
        body.send_keys(Keys.PAGE_DOWN)
        body.send_keys(Keys.ESCAPE)
        print("Scroll Count: {0}".format(i + 1))
        browser.implicitly_wait(1)

train, test = data_split(getData(), 80.0/100.0)

# transforme le texte en données numériques possibles à classifier (grâce au CountVectorizer (compteur d'occurences))
# on utilisera un classifieur "Support vector machine" qui est une généralisation de classifieur linéaire.
# puis un classifieur de Bayes


trainSentences = train[0]
trainSentiments = train[1]
testSentences = test[0]
testSentiments = test[1]

cv = CountVectorizer()
# crée un dictionnaire de vocabulaire des textes,
# sous forme d'une matrice comportant les termes et leurs fréquences.
cv.fit(trainSentences)

# Compte les occurences des mots de vocabulaire ("document term matrix")
trainSentences = cv.transform(trainSentences)
testSentences = cv.transform(testSentences)

# classifieur linéaire
svm = LinearSVC()
svm.fit(trainSentences, trainSentiments)


print("précision du modèle sur les données d'entrainement : {0}".format(svm.score(trainSentences, trainSentiments)))
print("précision du modèle sur les données de test : {0}".format(svm.score(testSentences, testSentiments)))

# affiche les résultats.
visualize_data(svm, cv.get_feature_names(), n_top_features=40)
plt.show()

# browse_facebook()
