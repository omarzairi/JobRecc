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

df= pd.concat([jobs1,jobs2,jobs3])
to_drop = ['JobDescription','JobRequirment','Combined']
df.drop(to_drop, inplace=True, axis=1)

classes = df['jobtitle'].value_counts()[:200]
keys = classes.keys().to_list()

df =df[df['jobtitle'].isin(keys)]
def chane_titles(x):
    x = x.strip()
    if x == 'Senior Java Developer':
        return 'Java Developer'
    elif x == 'Sr Java Developer':
        return 'Java Developer'
    elif x == 'Sr. Java Developer':
        return 'Java Developer'
    elif x == 'Senior Software Engineer':
        return 'Software Engineer'
    elif x == 'Senior QA Engineer':
        return 'Software QA Engineer'
    elif x == 'Senior Software Developer':
        return 'Senior Web Developer'
    elif x == 'Senior PHP Developer':
        return 'PHP Developer'
    elif x == 'Senior .NET Developer':
        return '.NET Developer'
    elif x == 'Sr .NET Developer':
        return '.NET Developer'
    elif x == 'Sr. .NET Developer':
        return '.NET Developer'
    elif x == '.Net Developer':
        return '.NET Developer'
    elif x == 'Senior Web Developer':
        return 'Web Developer'
    elif x == 'Database Administrator':
        return 'Database Admin/Dev'
    elif x == 'Database Developer':
        return 'Database Admin/Dev'

    else:
        return x


df['jobtitle'] = df['jobtitle'].apply(chane_titles)

stopwordsSkills=[]
sk=df["skills"]
for word in sk:
    word=str(word)
    word.lower()
    word=word.split(',')
    if(word[0]!=''):
        stopwordsSkills.append(word[0])
sdf=pd.DataFrame({'skills':stopwordsSkills})
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(df['jobtitle'].values)
analyze=vectorizer.build_analyzer()

#training the model
jobSkills=[]
for i in sdf["skills"]:
    jobSkills.append(i.lower())
Xclass=vectorizer.fit_transform(jobSkills)
X_train,X_test,Y_train,Y_test=train_test_split(Xclass,df['jobtitle'],test_size=0.2,random_state=42)
#predictions

svm=SVC(C=50,gamma=1,kernel='rbf',probability=True)
svmfit=svm.fit(X_train,Y_train)
svm_predictions=svmfit.predict(X_test)
def getJob(sk):
    skills=''
    for i in sk:
        i=i.lower()
        skills=skills+i+','
    pred = vectorizer.transform([skills])
    output = svm.predict(pred)
    return output[0]

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return "hello"
#return the getJob function in jason with skills given in body
@app.route('/getJob', methods=['POST'])
def getJobb():
    return getJob(request.get_json()["skills"])