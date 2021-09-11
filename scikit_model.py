import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
import joblib

count=CountVectorizer()

data=pd.read_csv("Train.csv")
data.head()

print(data.shape)

import re
def preprocessor(text):
             text=re.sub('<[^>]*>','',text)
             emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
             text=re.sub('[\W]+',' ',text.lower()) +\
                ' '.join(emojis).replace('-','')
             return text  

data['text']=data['text'].apply(preprocessor)

from nltk.stem.porter import PorterStemmer

porter=PorterStemmer()

def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop=stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True)
y=data.label.values
x=tfidf.fit_transform(data.text)
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)

import time
start_time = time.time()
from sklearn.linear_model import LogisticRegressionCV

clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("LogisticRegression Accuracy:",metrics.accuracy_score(y_test, y_pred))
joblib.dump(clf, "models/lr.sav")
print("Took " + str((time.time() - start_time)/60) + " mins")

from sklearn.linear_model import SGDClassifier
start_time = time.time()
clf= SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SGDClassifier Accuracy:",metrics.accuracy_score(y_test, y_pred))
joblib.dump(clf, "models/sgd.sav")
print("Took " + str((time.time() - start_time)/60) + " mins")

from sklearn.naive_bayes import MultinomialNB
start_time = time.time()
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, y_pred))
joblib.dump(clf, "models/nb.sav")
print("Took " + str((time.time() - start_time)/60) + " mins")

from sklearn.svm import LinearSVC
start_time = time.time()
clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("LinearSVC Accuracy:",metrics.accuracy_score(y_test, y_pred))
joblib.dump(clf, "models/svc.sav")
print("Took " + str((time.time() - start_time)/60) + " mins")

