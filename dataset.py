
import sys
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
from sklearn.multiclass import OneVsRestClassifier
#from trained_model import google_model, get_word_vector
from nltk.corpus import opinion_lexicon
from os import listdir
from os.path import isfile, join
import sklearn.cross_validation
import gensim, logging
from termcolor import colored
from colorama import init
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.parsing import PorterStemmer
global_stemmer = PorterStemmer()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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

X_train = []
y_train = []
X_test = []
y_test = []
X= []
y = []

class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """
 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
 
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
 
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word

def cleanSentences(string):
    string = string.lower()
    return re.sub(strip_special_chars, "", string.lower())


def makeDataSet():
	client = MongoClient(MONGO_HOST, MONGO_PORT)
	database = client["synopsis"]
	sessions = database["sessions"]
	global countFor
	global countAgainst
	global countDefect
	global countError
	global countAllegation
	global countAppreciation
	global countCallForAction
	for session in sessions.find():
		if "debates" in session.keys():
			for debate in session["debates"].keys():
				if debate == "5999644a37335cad52ecd861" or debate == "5999645137335cad52ecd862": # General
					for bill in session["debates"][debate]:		
						debateId = session["debates"][debate][bill]
						debates = database["debates"]
						debateBill = debates.find_one({'_id': ObjectId(debateId)})
 						data = {}
 						countMin = 0
						for speech in debateBill:
							if speech != "_id":
								countMin = countMin + 1
						if countMin >= 3:
							for speech in debateBill:
								if speech != "_id":
									data["stance"] = debateBill[speech]["stance"]
									data["speech"] = debateBill[speech]["speech"]
									data["name"] = debateBill[speech]["name"]
									#newlinespeech = cleanSentences(data["speech"])
									if "Defect" in data["stance"]:
										f = open("defect/" + str(debateId) + "_" + str(speech),"w+")
										f.write(data["speech"].encode('utf-8'))
										countDefect = countDefect + 1
									if "Allegation" in data["stance"]:
										f = open("allegation/" + str(debateId) + "_" + str(speech),"w+")
										f.write(data["speech"].encode('utf-8'))
										countAllegation = countAllegation + 1
									if "Appreciation" in data["stance"]:
										f = open("appreciation/" + str(debateId) + "_" + str(speech),"w+")
										f.write(data["speech"].encode('utf-8'))
										countAppreciation = countAppreciation + 1
									if "CallForAction" in data["stance"]:
										f = open("callforaction/" + str(debateId) + "_" + str(speech),"w+")
										f.write(data["speech"].encode('utf-8'))
										countCallForAction = countCallForAction + 1
									if "for" in data["stance"]:
										f = open("for/" + str(debateId) + "_" + str(speech),"w+")
										f.write(data["speech"].encode('utf-8'))
										countFor = countFor + 1
									if "against" in data["stance"]:
										f = open("against/" + str(debateId) + "_" + str(speech),"w+")
										f.write(data["speech"].encode('utf-8'))
										countAgainst = countAgainst + 1
				if debate == "5999646837335cad52ecd865" or debate == "59dce640a7401a6088699006" or debate == "5999643037335cad52ecd85e": # Calling Attention and Discussions related
						for bill in session["debates"][debate]:
							debateId = bill
							debates = database["debates"]
  							debateBill = debates.find_one({'_id': ObjectId(debateId)})

  	 						data = {}
							for speech in debateBill:
								if speech == "mattersMap":
										countMin = 0 
										for en in debateBill["mattersMap"]:
											countMin = countMin + 1
										if countMin >= 5:
											for en in debateBill["mattersMap"]:
												data["stance"] = debateBill[speech][en]["stance"]
												data["speech"] = debateBill[speech][en]["speech"]
												data["name"] = debateBill[speech][en]["name"]
												if "Defect" in data["stance"]:
													f = open("defect/" + str(debateId) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countDefect = countDefect + 1
												if "Allegation" in data["stance"]:
													f = open("allegation/" + str(debateId) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countAllegation = countAllegation + 1
												if "Appreciation" in data["stance"]:
													f = open("appreciation/" + str(debateId) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countAppreciation = countAppreciation + 1
												if "CallForAction" in data["stance"]:
													f = open("callforaction/" + str(debateId) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countCallForAction = countCallForAction + 1
												if "for" in data["stance"]:
													f = open("for/" + str(debateId) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countFor = countFor + 1
												if "against" in data["stance"]:
													f = open("against/" + str(debateId) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countAgainst = countAgainst + 1
				if debate == "5999649f37335cad52ecd86b": # Government Bills
						for bill in session["debates"][debate]:
							
							billId = session["debates"][debate][bill]["billId"]
							bill = session["debates"][debate][bill]["bill"]

							debates = database["debates"]
							bills = database["bills"]
					
							title = bills.find_one({'_id': ObjectId(billId)})["name"]
  							debateBill = debates.find_one({'_id': ObjectId(bill)})

  	 						data = {}
  	 						countMin = 0
							for speech in debateBill: 
								if speech != "_id":
									countMin = countMin + 1
							if countMin >= 3:
								for speech in debateBill: 
									if speech != "_id":
										if str(bill) == "5a42535389d32a43eb04d2bd":
											print "lulll"
										data["stance"] = debateBill[speech]["stance"]
										data["speech"] = debateBill[speech]["speech"]
										data["name"] = debateBill[speech]["name"]
										newlinespeech = cleanSentences(data["speech"])
										newline = ""
										if "Defect" in data["stance"]:
											f = open("defect/" + str(bill) + "_" + str(speech),"w+")
											f.write(data["speech"].encode('utf-8'))
											countDefect = countDefect + 1
										if "Allegation" in data["stance"]:
											f = open("allegation/" + str(bill) + "_" + str(speech),"w+")
											f.write(data["speech"].encode('utf-8'))
											countAllegation = countAllegation + 1
										if "Appreciation" in data["stance"]:
											f = open("appreciation/" + str(bill) + "_" + str(speech),"w+")
											f.write(data["speech"].encode('utf-8'))
											countAppreciation = countAppreciation + 1
										if "CallForAction" in data["stance"]:
											f = open("callforaction/" + str(bill) + "_" + str(speech),"w+")
											f.write(data["speech"].encode('utf-8'))
											countCallForAction = countCallForAction + 1
										if "for" in data["stance"]:
											f = open("for/" + str(bill) + "_" + str(speech),"w+")
											f.write(data["speech"].encode('utf-8'))
											countFor = countFor + 1
										if "against" in data["stance"]:
											f = open("against/" + str(bill) + "_" + str(speech),"w+")
											f.write(data["speech"].encode('utf-8'))
											countAgainst = countAgainst + 1
				if debate == "599965fa37335cad52ecd88f": # Statutory Resolutions
						for bill in session["debates"][debate]:
		
							debateId = session["debates"][debate][bill]
							debates = database["debates"]
  							debateBill = debates.find_one({'_id': ObjectId(debateId)})

  	 						data = {}
							for speech in debateBill:
								if speech == "mattersMap":
									countMin = 0
									for en in debateBill["mattersMap"]:
										countMin = countMin + 1
									if countMin >= 3:
										for en in debateBill["mattersMap"]:
											if str(debateId) == "5a42535389d32a43eb04d2bd":
												print "lullll"	
											data["stance"] = debateBill[speech][en]["stance"]
											data["speech"] = debateBill[speech][en]["speech"]
											data["name"] = debateBill[speech][en]["name"]
											newlinespeech = cleanSentences(data["speech"])
											newline = ""
											found = False
											if "Defect" in data["stance"]:
												f = open("defect/" + str(bill) + "_" + str(speech),"w+")
												f.write(data["speech"].encode('utf-8'))
												countDefect = countDefect + 1
											if "Allegation" in data["stance"]:
												f = open("allegation/" + str(bill) + "_" + str(speech),"w+")
												f.write(data["speech"].encode('utf-8'))
												countAllegation = countAllegation + 1
											if "Appreciation" in data["stance"]:
												f = open("appreciation/" + str(bill) + "_" + str(speech),"w+")
												f.write(data["speech"].encode('utf-8'))
												countAppreciation = countAppreciation + 1
											if "CallForAction" in data["stance"]:
												f = open("callforaction/" + str(bill) + "_" + str(speech),"w+")
												f.write(data["speech"].encode('utf-8'))
												countCallForAction = countCallForAction + 1
											if "for" in data["stance"]:
												f = open("for/" + str(bill) + "_" + str(speech),"w+")
												f.write(data["speech"].encode('utf-8'))
												countFor = countFor + 1
											if "against" in data["stance"]:
												f = open("against/" + str(bill) + "_" + str(speech),"w+")
												f.write(data["speech"].encode('utf-8'))
												countAgainst = countAgainst + 1
				if debate == "59d0d7f12589e39d8102872e" or debate == "5999659437335cad52ecd883" or debate == "5999660437335cad52ecd890": # Private Member Bill # Submission by members # Private Resoltion
						for bill in session["debates"][debate]:
							debateArray = session["debates"][debate][bill]
							for debateE in debateArray:
								debates = database["debates"]
  								debateBill = debates.find_one({'_id': ObjectId(debateE)})
  	 							data = {}	
								for speech in debateBill:
									if speech == "mattersMap":
										countMin = 0
										for en in debateBill["mattersMap"]:
											countMin = countMin + 1
										if countMin >= 3:
											for en in debateBill["mattersMap"]:
												if str(debateE) == "5a42535389d32a43eb04d2bd":
													print "lulllll"
												data["stance"] = debateBill[speech][en]["stance"]
												data["speech"] = debateBill[speech][en]["speech"]
												data["name"] = debateBill[speech][en]["name"]
												newlinespeech = cleanSentences(data["speech"])
												if "Defect" in data["stance"]:
													f = open("defect/" + str(debateE) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countDefect = countDefect + 1
												if "Allegation" in data["stance"]:
													f = open("allegation/" + str(debateE) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countAllegation = countAllegation + 1
												if "Appreciation" in data["stance"]:
													f = open("appreciation/" + str(debateE) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countAppreciation = countAppreciation + 1
												if "CallForAction" in data["stance"]:
													f = open("callforaction/" + str(debateE) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countCallForAction = countCallForAction + 1
												if "for" in data["stance"]:
													f = open("for/" + str(debateE) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countFor = countFor + 1
												if "against" in data["stance"]:
													f = open("against/" + str(debateE) + "_" + str(en),"w+")
													f.write(data["speech"].encode('utf-8'))
													countAgainst = countAgainst + 1
	f.close()
	print "Count For " + str(countFor)
	print "Count Against " + str(countAgainst)
	print "Count Defect " + str(countDefect)
	print "Count Allegation " + str(countAllegation)
	print "Count Appreciation " + str(countAppreciation)
	print "Count CallForAction " + str(countCallForAction)

	print "Count Fake For " + str(countFor*5)
	print "Count Fake Against " + str(countAgainst*5)
	print "Count Fake Defect " + str(countDefect*5)
	print "Count Fake Allegation " + str(countAllegation*5)
	print "Count Fake Appreciation " + str(countAppreciation*5)
	print "Count Fake CallForAction " + str(countCallForAction*5)

def getFeatureVector(speech):
	pass
	# Google Word2Vec
	'''speech = re.sub(' +',' ',speech)
	words = speech.split()
	google_vector = np.zeros(300, dtype='float64')
	for word in words:
		google_vector += get_word_vector(word)
	if np.all(google_vector == 0): return None
	return google_vector

	# Discourse Connectives
	discourseConnectives = ['above all', 'accordingly', 'actually', 'admittedly','after', 'after', 'after all', 'after that', 'afterwards', 'again', 'all in all', 'all the same', 'also', 'alternatively', 'although', 'always assuming that', 'and', 'or', 'anyway', 'as', 'as a consequence', 'as a corollary', 'as a result', 'as long as', 'as soon as', 'as well', 'at any rate', 'at first', 'at first sight', 'at first blush', 'at first view', 'at the moment when', 'at the outset', 'at the same time', 'because', 'before', 'but', 'by comparison', 'by contrast', 'by the same token', 'by the way', 'certainly', 'clearly', 'consequently', 'conversely', 'correspondingly', 'despite that', 'despite the fact that', 'earlier', 'either', 'else', 'equally', 'essentially then', 'even', 'even so', 'even then', 'eventually', 'every time', 'except', 'except insofar', 'finally', 'first', 'first of all', 'firstly', 'for', 'for a start', 'for example', 'for instance', 'for one thing', 'for the simple reason', 'for this reason', 'further', 'furthermore','further', 'given that', 'hence', 'however', 'if', 'if ever', 'if not', 'if only', 'if so', 'in a different vein', 'in actual fact','in addition', 'in any case', 'in case', 'in conclusion', 'in contrast','in fact', 'initially', 'in other words', 'in particular', 'in short', 'in spite of that', 'in sum', 'in that case', 'in the beginning', 'in the case of X','in the end','in the first place','in the meantime','in this way', 'in turn', 'in as much as','incidentally','indeed','instead','it follows that','it might appear that','it might seem that', 'just as','last', 'lastly','later','let us assume','likewise','meanwhile', 'merely','merely because','moreXly','moreover','mostly','much later','much sooner','naturally','neither is it the same','nevertheless','next','no doubt','nonetheless','not','not because','not only','not that','notably','notwithstanding that','notwithstanding that','now','now that','obviously','of course','on condition that','one one hand','on one side','on the assumption that','on the contrary','on the grounds that', 'on the one hand','on the one side','on the other hand','on the other side','once','once again','once more','or','or else','otherwise','overall','plainly','presumbly because','previously','provided that','providing that','put another way','rather','reciprocally','regardless of that','simply because','secondly','still','so that','since','similarly','simultaneously','so','specifically','still','subsequently','summarising','suppose','supposing that','surely','sure enough','such that','summing up','surely','that is','that is to say','the fact is that','the more often','then','thereafter','then again','therefore','thirdly','this time','thereby','this time','thus','though','to be sure','to conclude','to sum up','to start with','to begin with','thus','to the degree that','too','ultimetly','unless','we might say','when','wherein','while','yet','whenever','wheras','what is more','until','undoubtedly','true']
	discourse_vector = np.zeros(len(discourseConnectives), dtype='float64')
	for index, connective in enumerate(discourseConnectives):
		if connective in speech:
			discourse_vector[index] = 1
		else:
			discourse_vector[index] = 0
	#final_vector = np.append(google_vector, discourse_vector, axis=0)
	#return final_vector

	# Opinion Lexicons Count
	opinion_vector = np.zeros(1, dtype='float64')
	words = speech.split()
	countPositive = 0
	countNegative = 0
	for word in words:
		if word in opinion_lexicon.positive():
			countPositive = countPositive + 1
		elif word in opinion_lexicon.negative():
			countNegative = countNegative + 1
	if countPositive > countNegative:
		opinion_vector[0] = 1
	elif countPositive < countNegative:
		opinion_vector[0] = -1
	else:
		opinion_vector[0] = 0
	print opinion_vector
	return opinion_vector'''

	# Verbs

	# doc2vec





def makeFeatures():
	global X_train
	global X_test
	global y_train
	global y_test
	# CallforAction : 0, Defect : 1, Allegation : 2, Appreciation : 3
	CallforActionFiles = ['callforaction/' + f for f in listdir('callforaction/') if isfile(join('callforaction/', f))]
	count = 0
	for fi in CallforActionFiles:
		with open(fi, "r") as f:
			text=f.read()
			vector = getFeatureVector(text)
			if np.all(vector == 0):
				pass
			else:
				if count < (countCallForAction*(float(8)/10)):
					X_train.append(vector)
					y_train.append(1)
				else:
					X_test.append(vector)
					y_test.append(1)
				count = count + 1

	count = 0
	DefectFiles = ['defect/' + f for f in listdir('defect/') if isfile(join('defect/', f))]
	for fi in DefectFiles:
		with open(fi, "r") as f:
			text=f.read()
			vector = getFeatureVector(text)
			if np.all(vector == 0):
				pass
			else:
				if count < (countDefect*(float(8)/10)):
					X_train.append(vector)
					y_train.append(0)
				else:
					X_test.append(vector)
					y_test.append(0)
				count = count + 1

	count = 0
	AllegationFiles = ['allegation/' + f for f in listdir('allegation/') if isfile(join('allegation/', f))]
	for fi in AllegationFiles:
		with open(fi, "r") as f:
			text=f.read()
			vector = getFeatureVector(text)
			if np.all(vector == 0):
				pass
			else:
				if count < (countAllegation*(float(8)/10)):
					X_train.append(vector)
					y_train.append(1)
				else:
					X_test.append(vector)
					y_test.append(1)
				count = count + 1

	count = 0
	AppreciationFiles = ['appreciation/' + f for f in listdir('appreciation/') if isfile(join('appreciation/', f))]
	for fi in AppreciationFiles:
		with open(fi, "r") as f:
			text=f.read()
			vector = getFeatureVector(text)
			if np.all(vector == 0):
				pass
			else:
				if count < (countAppreciation*(float(8)/10)):
					X_train.append(vector)
					y_train.append(1)
				else:
					X_test.append(vector)
					y_test.append(1)
				count = count + 1

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

def makeModels():
	global X_train
	global X_test
	global y_train
	global y_test
	global X
	global y
	with open("X_train.p", "rb") as f:
		X_train = pickle.load(f)

	with open("y_train.p", "rb") as f:
		y_train = pickle.load(f)

	with open("X_test.p", "rb") as f:
		X_test = pickle.load(f)

	with open("y_test.p", "rb") as f:
		y_test = pickle.load(f)

	X = np.append(X_train, X_test, axis=0) 
	y = np.append(y_train, y_test)

	print X.shape
	print y.shape

	svmModel = OneVsRestClassifier(svm.LinearSVC())
	svmModel.fit(X_train, y_train)
	with open("svmModel.p", "wb") as f:
		pickle.dump(svmModel, f)

def makeResults():
	global X_train
	global X_test
	global y_train
	global y_test
	global X
	global y
	svmModel = None
	with open("svmModel.p", "rb") as f:
		svmModel = pickle.load(f)
	scores = cross_validation(X, y, svmModel, cv=5)
	pretty_print_scores(scores)

def cross_validation(data, target, classifier, cv=5):
    """
    Does a cross validation with the classifier
    parameters:
        - `data`: array-like, shape=[n_samples, n_features]
            Training vectors
        - `target`: array-like, shape=[n_samples]
            Target values for corresponding training vectors
        - `classifier`: A classifier from the scikit-learn family would work!
        - `cv`: number of times to do the cross validation. (default=5)
    return a list of numbers, where the length of the list is equal to `cv` argument.
    """
    return sklearn.cross_validation.cross_val_score(classifier, data, target, cv=cv)


def pretty_print_scores(scores):
    """
    Prints mean and std of a list of scores, pretty and colorful!
    parameter `scores` is a list of numbers.
    """
    print colored("                                      ", 'white', 'on_white')
    print colored(" Mean accuracy: %0.3f (+/- %0.3f std) " % (scores.mean(), scores.std() / 2), 'magenta', 'on_white', attrs=['bold'])
    print colored("                                      ", 'white', 'on_white')

def trainWord2Vec():
	words = []
	CallforActionFiles = ['callforaction/' + f for f in listdir('callforaction/') if isfile(join('callforaction/', f))]
	for fi in CallforActionFiles:
		with open(fi, "r") as f:
			text=f.read()
			sentenceList = sent_tokenize(text.decode('utf-8').strip())
			for sent in sentenceList:
				wordsList = word_tokenize(sent)
				newWordsList = []
				for wor in wordsList:
					newWordsList.append(StemmingHelper.stem(wor))
				words.append(newWordsList)

	DefectFiles = ['defect/' + f for f in listdir('defect/') if isfile(join('defect/', f))]
	for fi in DefectFiles:
		with open(fi, "r") as f:
			text=f.read()
			sentenceList = sent_tokenize(text.decode('utf-8').strip())
			for sent in sentenceList:
				wordsList = word_tokenize(sent)
				newWordsList = []
				for wor in wordsList:
					newWordsList.append(StemmingHelper.stem(wor))
				words.append(newWordsList)

	AllegationFiles = ['allegation/' + f for f in listdir('allegation/') if isfile(join('allegation/', f))]
	for fi in AllegationFiles:
		with open(fi, "r") as f:
			text=f.read()
			sentenceList = sent_tokenize(text.decode('utf-8').strip())
			for sent in sentenceList:
				wordsList = word_tokenize(sent)
				newWordsList = []
				for wor in wordsList:
					newWordsList.append(StemmingHelper.stem(wor))
				words.append(newWordsList)

	AppreciationFiles = ['appreciation/' + f for f in listdir('appreciation/') if isfile(join('appreciation/', f))]
	for fi in AppreciationFiles:
		with open(fi, "r") as f:
			text=f.read()
			sentenceList = sent_tokenize(text.decode('utf-8').strip())
			for sent in sentenceList:
				wordsList = word_tokenize(sent)
				newWordsList = []
				for wor in wordsList:
					newWordsList.append(StemmingHelper.stem(wor))
				words.append(newWordsList)

	print len(words)
	model = gensim.models.Word2Vec(words, min_count=1, workers=12)
	model.save('word2vec')
	words = list(model.wv.vocab)
	print(words)
	return model


if __name__ == "__main__":
	w2vpre = trainWord2Vec()
	#makeDataSet()
	#makeFeatures()
	#makeModels()
	#makeResults()

