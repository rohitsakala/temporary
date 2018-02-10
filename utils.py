import os
import re
from string import punctuation, whitespace

import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords as st
from stop_words import get_stop_words

LOT_OF_STOPWORDS = frozenset(list(STOPWORDS) + get_stop_words('en') + st.words('english'))

#Folders
MODELS = os.environ.get("MODELS", "models")
ENGLISHWORDS = os.environ.get("ENGLISHWORDS", "english-words")
VERBS = os.environ.get("VERBS", "Verbs")
DATA = os.environ.get('DATA', 'data')

#Files
GOOGLEMODEL = os.environ.get("GOOGLEMODEL", "GoogleNews-vectors-negative300.bin.gz")
WORDSFILE = os.environ.get("WORDSFILE", "big.txt")
BETH_LEVIN_VERBS = os.environ.get("BETH_LEVIN_VERBS", "beth_levin_person.json")
BETH_LEVIN_VERBS1 = os.environ.get("BETH_LEVIN_VERBS1", "verbs_levis.json")
BETH_LEVIN_SAMPLE = os.environ.get("BETH_LEVIN_SAMPLE", "new_classification.csv")
PRONOUNCEMENTS = os.environ.get('PRONOUNCEMENTS', 'Pronouncements.json')
ENGAGE = os.environ.get('ENGAGE', 'Engage.json')
RESPOND = os.environ.get('RESPOND', 'Respond.json')
FORCE = os.environ.get('FORCE', 'Force.json')

#NLP Tools
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
WHITE_PUNC_REGEX = re.compile(r"[%s]+" % re.escape(whitespace + punctuation), re.UNICODE)

#Utility Functions
def remove_stop_words(phrase):
    return '_'.join(list(set(map(lambda x: x.lower(), phrase.split('_'))) - LOT_OF_STOPWORDS))


def removing_characters_whitespaces(text):
    return ''.join(re.split(WHITE_PUNC_REGEX, text))
