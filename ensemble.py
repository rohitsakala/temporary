import os
import sys
import json
import nltk
import pickle
import itertools
import numpy as np
import multiprocessing

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from dataset import countFor,countAgainst,countDefect,countError,countAllegation,countAppreciation,countCallForAction

from features_spacy import getVector
from ensemble_classifier import EnsembleClassifier

X_train = []
y_train = []
X_test = []
y_test = []

def get_total_vector(sent):
    ans = getVector(sent)
    return ans

print "Extracting Data"
# CallforAction : 0, Defect : 1, Allegation : 2, Appreciation : 3

print "CallforAction Training Data Processing"
CallforActionFiles = ['callforaction/' + f for f in listdir('callforaction/') if isfile(join('callforaction/', f))]
count = 0
countChu = 0
for fi in CallforActionFiles:
        with open(fi, "r") as f:
            text=f.read()
            splittedtext = text.split()
            ans = get_total_vector(splittedtext)
            if np.all(ans == 0):
                countChu = countChu + 1
            else:
                if count < (countCallForAction*(float(8)/10)):
                    X_train.append(ans)
                    y_train.append(0)
                else:
                    X_test.append(ans)
                    y_test.append(0)
                count = count + 1


print len(X_train)
print len(y_train)
print len(X_test)
print len(y_test)

count = 0
print "Defect Training Data Processing"
DefectFiles = ['defect/' + f for f in listdir('defect/') if isfile(join('defect/', f))]
for fi in DefectFiles:
        with open(fi, "r") as f:
            text=f.read()
            splittedtext = text.split()
            ans = get_total_vector(splittedtext)
            if np.all(ans == 0):
                countChu = countChu + 1
            else:
                if count < (countDefect*(float(8)/10)):
                    X_train.append(ans)
                    y_train.append(1)
                else:
                    X_test.append(ans)
                    y_test.append(1)
                count = count + 1

print len(X_train)
print len(y_train)
print len(X_test)
print len(y_test)

count = 0
print "Allegation Training Data Processing"
AllegationFiles = ['allegation/' + f for f in listdir('allegation/') if isfile(join('allegation/', f))]
for fi in AllegationFiles:
        with open(fi, "r") as f:
            text=f.read()
            splittedtext = text.split()
            ans = get_total_vector(splittedtext)
            if np.all(ans == 0):
                countChu = countChu + 1
            else:
                if count < (countAllegation*(float(8)/10)):
                    X_train.append(ans)
                    y_train.append(2)
                else:
                    X_test.append(ans)
                    y_test.append(2)
                count = count + 1

print len(X_train)
print len(y_train)
print len(X_test)
print len(y_test)

count = 0
print "Appreciation Training Data Processing"
AppreciationFiles = ['appreciation/' + f for f in listdir('appreciation/') if isfile(join('appreciation/', f))]
for fi in AppreciationFiles:
        with open(fi, "r") as f:
            text=f.read()
            splittedtext = text.split()
            ans = get_total_vector(splittedtext)
            if np.all(ans == 0):
                countChu = countChu + 1
            else:
                if count < (countAppreciation*(float(8)/10)):
                    X_train.append(ans)
                    y_train.append(3)
                else:
                    X_test.append(ans)
                    y_test.append(3)
                count = count + 1

print len(X_train)
print len(y_train)
print len(X_test)
print len(y_test)

print countChu

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape


with open("X_train.p", "wb") as f:
    pickle.dump(X_train, f)

with open("y_train.p", "wb") as f:
    pickle.dump(y_train, f)

with open("X_test.p", "wb") as f:
    pickle.dump(X_test, f)

with open("y_test.p", "wb") as f:
    pickle.dump(y_test, f)
'''

X_train  = None
y_train = None
X_test = None
y_test = None
with open("X_train.p", "rb") as f:
    X_train = pickle.load(f)

with open("y_train.p", "rb") as f:
    y_train = pickle.load(f)

with open("X_test.p", "rb") as f:
    X_test = pickle.load(f)

with open("y_test.p", "rb") as f:
    y_test = pickle.load(f)
'''
X = np.append(X_train, X_test, axis=0) 
y = np.append(y_train, y_test)

print X.shape
print y.shape

print "Training Model"
#svmmodel = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
clf1 = OneVsRestClassifier(LogisticRegression(n_jobs=10))
clf2 = OneVsRestClassifier(RandomForestClassifier(n_jobs=10))
clf3 = OneVsRestClassifier(GaussianNB())

eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])


import autosklearn.classification
import sklearn.model_selection
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

'''
global scores

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):

    print "*****************************************************"
    print "                    ", label, "                     "
    print "*****************************************************"
    scores = cross_validate(clf, X, y, scoring='accuracy', return_train_score=True)
    #sum_sc = 0
    for key in scores.keys():
        print key, scores[key]
    #    sum_sc = sum_sc + scores[key]
    #print "Accuracy: " + str(float(sum_sc)/10)
    scores = cross_validate(clf, X, y,  scoring='recall',  return_train_score=True)
    for key in scores.keys():
        print key, scores[key]
    #    sum_sc = sum_sc + scores[key]
    #print "Accuracy: " + str(float(sum_sc)/10)
    scores = cross_validate(clf, X, y,  scoring='precision',  return_train_score=True)
    for key in scores.keys():
        print key, scores[key]
    #    sum_sc = sum_sc + scores[key]
    #print "Accuracy: " + str(float(sum_sc)/10)
    scores = cross_validate(clf, X, y,  scoring='f1',  return_train_score=True)
    for key in scores.keys():
        print key, scores[key]
    #    sum_sc = sum_sc + scores[key]
    #print "Accuracy: " + str(float(sum_sc)/10)
    predicted = cross_val_predict(clf, X, y, cv=10)
    print metrics.accuracy_score(y, predicted)
'''


"""
print "Building Model"
svmmodel.fit(X_train, y_train.ravel())
print "Completed Building Model"
print(svmmodel.feature_importances_)
with open("svm_model.p", "wb") as f:
    pickle.dump(svmmodel, f)

total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(X_train), y_train))))

print "This Accuracy"
print sum(total) / float(len(total))
total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(X_test), y_test))))

print "That Accuracy"
print sum(total) / float(len(total))
"""
