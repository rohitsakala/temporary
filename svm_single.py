import os
import sys
import json
import nltk
import pickle
import itertools
import numpy as np
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from dataset import PRONOUNCEMENTS_TRAIN_DATA, PRONOUNCEMENTS_TEST_DATA, \
                    ENGAGE_TRAIN_DATA, ENGAGE_TEST_DATA, \
                    RESPOND_TRAIN_DATA, RESPOND_TEST_DATA, \
                    FORCE_TRAIN_DATA, FORCE_TEST_DATA

from features_spacy import get_vector


X_train = []
y_train = []
X_test = []
y_test = []
TRAINING_LIMIT = 120000000
TESTING_LIMIT = 30000000

def get_total_vector(sent):
    #try:
    ans = get_vector(sent)
    #print len(ans)
    #except:
    #    return None
    return ans

print "Extracting Data"
# PRONOUNCEMENTS : 0, ENGAGE : 1, RESPOND : 2, FORCE : 3

print "Pronouncements Training Data Processing"
count = 0
for sent in PRONOUNCEMENTS_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_train.append(ans)
        y_train.append(0)
    count += 1

count = 0
print "Pronouncements Training Data Processing Done"
print "Engage Training Data Processing"
for sent in ENGAGE_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_train.append(ans)
        y_train.append(1)
    count += 1

count = 0
print "Engage Training Data Processing Done"
print "Respond Training Data Processing"
for sent in RESPOND_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_train.append(ans)
        y_train.append(2)
    count += 1

count = 0
print "Respond Training Data Processing Done"
print "Force Training Data Processing"
for sent in FORCE_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_train.append(ans)
        y_train.append(3)
    count += 1

print "Force Training Data Processing Done"
count = 0
print "Pronouncements Testing Data Processing"
for sent in PRONOUNCEMENTS_TEST_DATA:
    if count > TESTING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_test.append(ans)
        y_test.append(0)
    count += 1


print "Pronouncements Testing Data Processing Done"
count = 0
print "Engage Testing Data Processing"
for sent in ENGAGE_TEST_DATA:
    if count > TESTING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_test.append(ans)
        y_test.append(1)
    count += 1

print "Engage Testing Data Processing Done"
count = 0
results = []
print "Respond Testing Data Processing"
for sent in RESPOND_TEST_DATA:
    if count > TESTING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_test.append(ans)
        y_test.append(2)
    count += 1
print "Respond Testing Data Processing Done"

count = 0
print "Force Testing Data Processing"
for sent in FORCE_TEST_DATA:
    if count > TESTING_LIMIT: break
    ans = get_total_vector(sent)
    if ans is not None:
        #print count
        X_test.append(ans)
        y_test.append(3)
    count += 1
print "Force Testing Data Processing Done"

#for i, j in itertools.izip(X_train, y_train):
#    print type(i), i
#    print len(i), j
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print "Training Model"
#svmmodel = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
svmmodel = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1))
print "Building Model"
svmmodel.fit(X_train, y_train.ravel())
print "Completed Building Model"
#print(svmmodel.feature_importances_)
with open("svm_model.p", "wb") as f:
    pickle.dump(svmmodel, f)

total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(X_train), y_train))))

print "This Accuracy"
print sum(total) / float(len(total))
total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(X_test), y_test))))

print "That Accuracy"
print sum(total) / float(len(total))

