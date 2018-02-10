import os
import sys
import json
import numpy
import csv
reload(sys)
sys.setdefaultencoding('utf-8')
from trained_model import google_model, get_word_vector
from utils import VERBS, BETH_LEVIN_VERBS, BETH_LEVIN_VERBS1, lemmatizer, BETH_LEVIN_SAMPLE

import spacy
SPACY_NLP = spacy.load('en')
DEPENDENCIES = {'acl': 19, 'advcl': 10,  'advmod': 11,  'amod': 20,  'appos': 17, \
                    'aux': 13,  'case': 23,  'cc': 25,  'ccomp': 4,  'clf': 22, \
                    'compound': 28, 'conj': 24, 'cop': 14, 'csubj': 3, 'dep': 36, \
                    'det': 21,  'discourse': 12,  'dislocated': 9,  'expl': 8, \
                    'fixed': 26, 'flat': 27, 'goeswith': 32, 'iobj': 2, 'list': 29, \
                    'mark': 15, 'nmod': 16, 'nsubj': 0,  'nummod': 18, 'obj': 1, \
                    'obl': 6, 'orphan': 31, 'parataxis': 30, 'punct': 34, 'reparandum': 33, \
                    'root': 35, 'vocative': 7, 'xcomp': 5}

#DEPENDENCIES = {'advcl': 12, 'advmod': 2,'amod': 6, 'aux': 13, 'ccomp': 3, 'cop': 7, 'csubj': 5, 'expl': 4, 'iobj': 11, \
#                 'mark': 10, 'nmod': 9, 'nsubj': 1, 'obj': 8, 'parataxis': 0}


OUR_CLASSIFICATION_VERBS = []
fio = open(os.path.join(VERBS, BETH_LEVIN_SAMPLE) , "rb")
rea = csv.reader(fio)
for row in rea:
    ids = [i.strip() for i in row[1:]]
    OUR_CLASSIFICATION_VERBS.extend(ids)

OUR_CLASSIFICATION_VERBS = list(set(OUR_CLASSIFICATION_VERBS))

verbs_classes = {}
with open(os.path.join(VERBS, BETH_LEVIN_VERBS1), "rb") as f:
    verbs_classes = json.load(f)

class_dict = {}
class_index = 0
for clas in verbs_classes.keys():
    if clas not in OUR_CLASSIFICATION_VERBS: continue
    class_dict[clas] = class_index
    class_index += 1
#print len(class_dict)
# [u'modifiers', u'word', u'NE', u'POS_coarse', u'lemma', u'arc', u'POS_fine']
def get_modifier_word_arc(lista):
    if type(lista) is list and len(lista) is 0: return
    elif len(lista['modifiers']) is 0:
        return [(lista['word'], lista['arc'])]
    else:
        ans = [(lista['word'], lista['arc'])]
        for modifier in lista['modifiers']:
            ans.extend(get_modifier_word_arc(modifier))
        return ans



def get_vector(sentence):
    global SPACY_NLP, DEPENDENCIES, verbs_classes
    text = SPACY_NLP(unicode(sentence.lower()))
    text = text.print_tree()
    return_vector = numpy.zeros(len(DEPENDENCIES), dtype='float64')
    classes_vector = numpy.zeros(len(class_dict.keys()), dtype='float64')
    google_vector = numpy.zeros(300, dtype='float64')
    for sent in text:
        ans = get_modifier_word_arc(sent)
        for tup in ans:
            if tup[1] == 'ROOT':
                lemmatised_word = lemmatizer.lemmatize(tup[0])
                for mclass in verbs_classes.keys():
                    if mclass not in class_dict.keys(): continue
                    if lemmatised_word.upper() in verbs_classes[mclass]:
                        classes_vector[class_dict[mclass]] += 1
            elif tup[1] in DEPENDENCIES.keys():
                return_vector[DEPENDENCIES[tup[1]]] += 1
            google_vector += get_word_vector(tup[0])
    #print "Google Vector: ", len(google_vector)
    #print "Classes Vector: ", len(classes_vector)
    #print "Return Vector: ", len(return_vector)
    ans = numpy.append(google_vector, numpy.append(return_vector, classes_vector))
    if numpy.all(ans == 0): return None
    return ans

def getVector(speech):
    words = speech.split()
    google_vector = numpy.zeros(300, dtype='float64')
    for word in words:
        google_vector += get_word_vector(word)
    print "Google Vector: ", len(google_vector)
    ans = google_vector
    if numpy.all(ans == 0): return None
    return ans
