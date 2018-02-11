import os
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import sys
from os import listdir
import numpy as np
import pickle
from os.path import isfile, join
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_recall_fscore_support 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
#from trained_model import google_model, get_word_vector
from nltk.corpus import opinion_lexicon
from os import listdir
from collections import *
from os.path import isfile, join
import gensim
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
countFor = 0
countAgainst = 0
countDefect = 0
countError = 0
countAllegation = 0
countAppreciation = 0
countCallForAction = 0

MONGO_HOST = "10.2.32.86"
MONGO_PORT = 27017

X = []
y = []

def makeData():
	global X
	global y
	CallforActionFiles = ['callforaction/' + f for f in listdir('callforaction/') if isfile(join('callforaction/', f))]
	for fi in CallforActionFiles:
		with open(fi, "r") as f:
			text=f.read()
			X.append(text.split())
			y.append(0)

	DefectFiles = ['defect/' + f for f in listdir('defect/') if isfile(join('defect/', f))]
	for fi in DefectFiles:
		with open(fi, "r") as f:
			text=f.read()
			X.append(text.split())
			y.append(1)

	AllegationFiles = ['allegation/' + f for f in listdir('allegation/') if isfile(join('allegation/', f))]
	for fi in AllegationFiles:
		with open(fi, "r") as f:
			text=f.read()
			X.append(text.split())
			y.append(2)

	AppreciationFiles = ['appreciation/' + f for f in listdir('appreciation/') if isfile(join('appreciation/', f))]
	for fi in AppreciationFiles:
		with open(fi, "r") as f:
			text=f.read()
			X.append(text.split())
			y.append(3)

	model = gensim.models.Word2Vec(X, size=100)
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	return w2v

def makeDataPre():
	with open("glove.6B.50d.txt", "rb") as lines:
		w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
			for line in lines}
	return w2v

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def makeModelss(w2v):
	global X
	global y
	svm_w2v = Pipeline([
    	("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    	("extra trees", OneVsRestClassifier(svm.LinearSVC()))])
	svm_w2v_tfidf = Pipeline([
    	("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    	("extra trees", OneVsRestClassifier(svm.LinearSVC()))])
	svm_w2v.fit(X,y)
	svm_w2v_tfidf.fit(X,y)
	print svm_w2v.score(X,y)
	print svm_w2v_tfidf.score(X,y)

if __name__ == "__main__":
	w2v = makeData()
	w2v_pretrained = makeDataPre()
	makeModelss(w2v)
	makeModelss(w2v_pretrained)
	#temp()
	