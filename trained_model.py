import os
import gensim
import numpy

from utils import MODELS, GOOGLEMODEL, lemmatizer

google_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(MODELS, GOOGLEMODEL), binary=True)

def get_word_vector(word):
    global google_model
    try:
        return google_model.word_vec(word.lower())
    except:
        try:
            return google_model.word_vec(lemmatizer.lemmatize(word.lower()))
        except:
            return numpy.zeros(300, dtype='float64')
