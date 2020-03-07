import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import sys

from sklearn.model_selection import train_test_split

import pickle

def load_data(csvFile):

    # read data
    df = pd.read_csv(csvFile)
        
    # encode data
    le = LabelEncoder()
    le.fit(df['class'])
    df['label'] = le.transform(df['class'])
    
    X, y = df['message'], df['label']

    # count vectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)

    # save encoder
    pickle.dump(cv, open('transform.pkl', 'wb'))

    return X, y

def train_model(X, y):

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # create model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    # save model
    filename = 'nlp_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))

if __name__ == "__main__":
    csvFile = sys.argv[1]
    print(csvFile)
    X, y = load_data(csvFile)
    train_model(X, y)