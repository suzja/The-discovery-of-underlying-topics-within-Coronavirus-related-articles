#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-,

import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import csv
from csv import reader
from nltk.tokenize import word_tokenize
import gensim
from gensim.utils import simple_preprocess
# os.chdir('..')
import gensim.corpora as corpora
from pprint import pprint

import pyLDAvis
import pyLDAvis.gensim_models
import pickle

import os
import spacy
from spacy.lang.nl.examples import sentences
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim import corpora, models

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def make_lda(name):
    words = []
    words.clear()
    stop_words = stopwords.words("dutch")
    stop_words.extend(['nrc', 'nrc.nl','nu', 'kun', 'nunl', 'ad', 'twitter', 'nu.nl', 'NU.nl', 'https', 'telegraaf', 'telegraaf.nl', 'the', 'to', 'volkskrant', 'volkskrant.nl', 'dagelijksestandaard'])
    # stop_words = (newStopWords)
    # print(stop_words)
    articles = pd.read_csv('../csvbestanden/' + str(name) + '.csv', sep="|", error_bad_lines=False)

    articles['text_processed'] = \
    articles['Text'].map(lambda x: re.sub('[,\.!?]', '', x))

    articles['text_processed'] = \
    articles['text_processed'].map(lambda x: x.lower())
    articles['text_processed'] = articles['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


    dedata = articles.text_processed.values.tolist()
    goededata = list(dedata)

    dedata_words = list(sent_to_words(dedata))

    nlp = spacy.load("nl_core_news_sm")
    excluded_tags = { "VERB", "ADP", "PUNCT", "NUM", "SYM", "AUX", "ADV", "CONJ", "DET", "PART", "PRON", "SCONJ", "X"}

    for i in goededata:
        nlpdata = nlp(str(i))
        for j in nlpdata:
            if ((j.pos_ not in excluded_tags) and (j.is_stop is False)):
                words.append(j)


    return(goededata)





if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn


    media = ["nudata", "nrcdata", "standaarddata", "volkskrantdata", "telegraafdata"]
    media1 = ["standaarddata"]

    for i in range(len(media1)):

        wordlist = make_lda(media1[i])
        poswords = list(sent_to_words(wordlist))
        print(wordlist)
