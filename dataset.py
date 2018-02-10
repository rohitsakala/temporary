import os
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import sys
from os import listdir
import numpy as np
from os.path import isfile, join
from trained_model import google_model, get_word_vector


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
	# Google Word2Vec
    words = speech.split()
    google_vector = np.zeros(300, dtype='float64')
    for word in words:
        google_vector += get_word_vector(word)
    print "Google Vector: ", len(google_vector)
    vector = google_vector
    if np.all(vector == 0): return None
    return vector

def makeFeatures():
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
					y_train.append(0)
				else:
					X_test.append(vector)
					y_test.append(0)
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
					y_train.append(1)
				else:
					X_test.append(vector)
					y_test.append(1)
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
					y_train.append(2)
				else:
					X_test.append(vector)
					y_test.append(2)
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
					y_train.append(3)
				else:
					X_test.append(vector)
					y_test.append(3)
				count = count + 1

	print len(X_train)
	print len(y_train)
	print len(X_test)
	print len(y_test)

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

if __name__ == "__main__":
	makeDataSet()
	makeFeatures()
