import os
import sys
import json
import nltk
import pickle
import itertools
import numpy as np
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from dataset import PRONOUNCEMENTS_TRAIN_DATA, PRONOUNCEMENTS_TEST_DATA, \
                    ENGAGE_TRAIN_DATA, ENGAGE_TEST_DATA, \
                    RESPOND_TRAIN_DATA, RESPOND_TEST_DATA, \
                    FORCE_TRAIN_DATA, FORCE_TEST_DATA

from features_spacy import get_vector


X_train = []
y_train = []
X_test = []
y_test = []
TRAINING_LIMIT = 100
TESTING_LIMIT = 20

def get_total_vector(sent):
    try:
        ans = get_vector(sent)
    except:
        return None
    print len(ans)
    return ans

pool_count = 40
pool = multiprocessing.Pool(pool_count)

print "Extracting Data"
# PRONOUNCEMENTS : 0, ENGAGE : 1, RESPOND : 2, FORCE : 3
print "Pronouncements Training Data Processing"
count = 0
results = []
for sent in PRONOUNCEMENTS_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    if count % pool_count == 0:
        print "Entering ans"
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_train.append(ans)
                y_train.append(0)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1

count = 0
results = []
print "Pronouncements Training Data Processing Done"
print "Engage Training Data Processing"
for sent in ENGAGE_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_train.append(ans)
                y_train.append(1)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1

count = 0
results = []
print "Engage Training Data Processing Done"
print "Respond Training Data Processing"
for sent in RESPOND_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_train.append(ans)
                y_train.append(2)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1

count = 0
results = []
print "Respond Training Data Processing Done"
print "Force Training Data Processing"
for sent in FORCE_TRAIN_DATA:
    if count > TRAINING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_train.append(ans)
                y_train.append(3)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1
print "Force Training Data Processing Done"

count = 0
results = []
print "Pronouncements Testing Data Processing"
for sent in PRONOUNCEMENTS_TEST_DATA:
    if count > TESTING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_test.append(ans)
                y_test.append(0)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1
print "Pronouncements Testing Data Processing Done"
count = 0
results = []
print "Engage Testing Data Processing"
for sent in ENGAGE_TEST_DATA:
    if count > TESTING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_test.append(ans)
                y_test.append(0)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1
print "Engage Testing Data Processing Done"
count = 0
results = []
print "Respond Testing Data Processing"
for sent in RESPOND_TEST_DATA:
    if count > TESTING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_test.append(ans)
                y_test.append(0)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
    count += 1
print "Respond Testing Data Processing Done"
count = 0
results = []
print "Force Testing Data Processing"
for sent in FORCE_TEST_DATA:
    if count > TESTING_LIMIT: break
    if count % pool_count == 0:
        for ans in results:
            ans = ans.get()
            if ans is not None:
                print count
                X_test.append(ans)
                y_test.append(0)
        results = []
    else:
        results.append(pool.apply_async(get_total_vector, (sent)))
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
svmmodel = SVC(kernel='linear')

print "Building Model"
svmmodel.fit(X_train, y_train.ravel())
print "Completed Building Model"

with open("svm_model.p", "wb") as f:
    pickle.dump(svmmodel, f)

total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(X_train), y_train))))

print "This Accuracy"
print sum(total) / float(len(total))
total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(X_test), y_test))))

print "That Accuracy"
print sum(total) / float(len(total))
