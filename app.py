from flask import Flask, render_template, url_for, request

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

# locations
model_path = "nlp_model.pkl"
cv_path = "transform.pkl"

# load objects
clf = pickle.load(open(model_path, 'rb'))
cv = pickle.load(open(cv_path, 'rb'))

# create Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        data = [request.form['message']]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)

    return render_template('result.html', prediction=prediction, input_text=request.form['message'])

if __name__ == "__main__":
    app.run(debug=True)