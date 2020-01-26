from flask import Flask, jsonify, render_template, request, redirect
import sqlalchemy as db
import random
from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize    
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import PorterStemmer
import string
import re

app = Flask(__name__)

engine = db.create_engine('sqlite:///posts.db')

metadata = db.MetaData()
posts = db.Table('posts', metadata, autoload=True, autoload_with=engine)

labels = ['beauty', 'business', 'cooking', 'education', 'hashtag', 'health',
       'information', 'job', 'link', 'miscellaneous', 'question',
       'relationship', 'salutation', 'self', 'travel']

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def punct_space(text):
    chars = []
    for ch in text:
        if(ch in string.punctuation):
            chars.append(' ')
        else:
            chars.append(ch)
    return "".join(chars)

def tokenize(text):
    text = punct_space(text)
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    stems = [token for token in stems if len(token) > 1] # Remove single letter tokens
    return stems

classifier = joblib.load("../models/type_detection_2019_07_14_pipe.sav")

def getProbabilities(post):
        probabilities = classifier.predict_proba([post])[0]
        vector = {}
        vector['PREDICTED'] = classifier.predict([post])[0]
        for i, j in zip(labels, probabilities):
                vector[i] = j*100
        return vector

def getPost():
        with engine.connect() as connection:
                query = "SELECT id, POST, COMMUNITY_NAME, COMMUNITY_TYPE FROM posts WHERE LABEL is null ORDER BY RANDOM() LIMIT 1"
                data = connection.execute(query).fetchall()
                return dict(data[0]) if data else None

def markPost(id, label):
        with engine.connect() as connection:
                query = f'UPDATE posts SET LABEL="{label}" WHERE id={id}'
                connection.execute(query)


def getProfile():
        with engine.connect() as connection:
                query = 'SELECT profile_picture_url,"index" FROM profiles WHERE label is null ORDER BY RANDOM() LIMIT 1'
                data = connection.execute(query).fetchall()
                print(dict(data[0]))
                return dict(data[0]) if data else None

def markProfile(index, label):
        with engine.connect() as connection:
                query = f'UPDATE profiles SET label="{label}" WHERE "index"={index}'
                connection.execute(query)

@app.route('/', methods=['GET', 'POST'])
def index():

        if request.method == 'GET':
                result = getPost()
                if(result is None):
                        return render_template("end.html")
                return render_template("index.html", result=result, prediction=getProbabilities(result['POST']))

        elif request.method == 'POST':
                if request.form['label'] == 'next':
                        return redirect("/")                

                markPost(request.form['id'], request.form['label'])
                return redirect("/")

@app.route('/dp', methods=['GET', 'POST'])
def dp():

        if request.method == 'GET':
                result = getProfile()
                if(result is None):
                        return render_template("end.html")
                return render_template("dp.html", result= result)

        elif request.method == 'POST':
                if request.form['label'] == 'next':
                        return redirect("/dp")                

                markProfile(request.form['id'], request.form['label'])
                return redirect("/dp")


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)