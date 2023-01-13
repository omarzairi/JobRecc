import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
from flask import Flask, request
from sklearn.metrics import classification_report
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
jobs1=pd.read_csv("./input/dataset/jobz.csv")
jobs2=pd.read_csv("./input/dataset/Modified.csv")
jobs3=pd.read_csv("./input/dataset/jobss.csv")
jobs2.rename(columns = {'Title':'jobtitle'}, inplace = True)
jobs2.rename(columns = {'RequiredQual':'skills'}, inplace = True)


app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return "hello"
#return the getJob function in jason with skills given in body
