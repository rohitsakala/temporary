import os
import sys
import json
import numpy
reload(sys)
sys.setdefaultencoding('utf-8')
from trained_model import google_model, get_word_vector
from utils import VERBS, BETH_LEVIN_VERBS, lemmatizer

from predpatt import PredPatt
DEPENDENCIES = {'acl': 19, 'advcl': 10,  'advmod': 11,  'amod': 20,  'appos': 17, \
                    'aux': 13,  'case': 23,  'cc': 25,  'ccomp': 4,  'clf': 22, \
                    'compound': 28, 'conj': 24, 'cop': 14, 'csubj': 3, 'dep': 36, \
                    'det': 21,  'discourse': 12,  'dislocated': 9,  'expl': 8, \
                    'fixed': 26, 'flat': 27, 'goeswith': 32, 'iobj': 2, 'list': 29, \
                    'mark': 15, 'nmod': 16, 'nsubj': 0,  'nummod': 18, 'obj': 1, \
                    'obl': 6, 'orphan': 31, 'parataxis': 30, 'punct': 34, 'reparandum': 33, \
                    'root': 35, 'vocative': 7, 'xcomp': 5}


verbs_classes = {}
with open(os.path.join(VERBS, BETH_LEVIN_VERBS), "rb") as f:
    verbs_classes = json.load(f)

class_dict = {}
class_index = 0
for clas in verbs_classes.keys():
    class_dict[clas] = class_index
    class_index += 1

def get_vector(sentence):
    global DEPENDENCIES, verbs_classes, class_index
    sent = PredPatt.from_sentence(sentence)
    #print sent.pprint()
    return_vector = numpy.zeros(len(DEPENDENCIES), dtype='float64')
    classes_vector = numpy.zeros(4, dtype='float64')
    google_vector = numpy.zeros(300, dtype='float64')
    for predicate in sent.events:
        #print "Predicate: ", predicate
        #print "Predicate Root Text: ", predicate.root.text
        lemmatised_word = lemmatizer.lemmatize(predicate.root.text.lower())
        for mclass in verbs_classes.keys():
            if lemmatised_word.upper() in verbs_classes[mclass]:
                classes_vector[class_dict[mclass]] += 1
        google_vector += get_word_vector(predicate.root.text)
        for argument in sent.argument_extract(predicate):
            #print "Argument: ", argument
            google_vector += get_word_vector(argument.root.text)
            for rule in argument.rules:
                #print "Rule: ", rule
                try: rule_name = rule.edge
                except: continue
                #print "Rule Name: ", rule_name
                try: return_vector[DEPENDENCIES[rule_name.rel]] += 1
                except: pass
    #print "Google Vector: ", len(google_vector)
    #print "Classes Vector: ", len(classes_vector)
    #print "Return Vector: ", len(return_vector)
    ans = numpy.append(google_vector, numpy.append(return_vector, classes_vector))
    if numpy.all(ans == 0): return None
    return ans
