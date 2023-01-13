import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
from flask import Flask, render_template, request




app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return "hello"
#return the getJob function in jason with skills given in body
