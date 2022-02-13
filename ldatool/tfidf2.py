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
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def make_lda(name):
    words = []
    words.clear()
    stop_words = stopwords.words("dutch")
    stop_words.extend(['nrc', 'nrc.nl','nu', 'kun', 'nunl', 'ad', 'twitter', 'nu.nl', 'NU.nl', 'https', 'telegraaf', 'telegraaf.nl', 'the', 'to', 'volkskrant', 'volkskrant.nl', 'dagelijksestandaard', 'én', 'één', 'óók'])
    # stop_words = (newStopWords)
    # print(stop_words)
    articles = pd.read_csv('../csvbestanden/' + str(name) + '.csv', sep="|", error_bad_lines=False)

    articles['text_processed'] = \
    articles['Text'].map(lambda x: re.sub('[,\.!?]', '', x))

    articles['text_processed'] = \
    articles['text_processed'].map(lambda x: x.lower())
    articles['text_processed'] = articles['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    articles['text_processed'] = articles['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if not word.isdigit()]))


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

        wordlist = make_lda(media[i])
        poswords = list(sent_to_words(wordlist))
        # print(poswords[:1][0][:30])

        dictword = corpora.Dictionary(poswords)
        texts = poswords
        corpus = [dictword.doc2bow(text) for text in texts]
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(corpus)
        # vectorizer.get_feature_names_out()
        # print(corpus[:1][0][:30])
        # print(corpus)

        #instantiate CountVectorizer()
        cv=CountVectorizer()

        # this steps generates word counts for the words in your docs
        word_count_vector=cv.fit_transform(wordlist)
        # print(word_count_vector.shape)

        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        # print idf values
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

        # sort ascending
        dfsort = df_idf.sort_values(by=['idf_weights'])
        # print(dfsort)

        # count matrix
        count_vector=cv.transform(wordlist)

        # tf-idf scores
        tf_idf_vector=tfidf_transformer.transform(count_vector)

        feature_names = cv.get_feature_names()

        #get tfidf vector for first document
        first_document_vector=tf_idf_vector[0]

        #print the scores
        df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
        dfs = df.sort_values(by=["tfidf"],ascending=False)
        print(dfs)
        #
        delijst = dfs["tfidf"].tolist()
        # print(delijst)
        deindex = dfs.index.values.tolist()
        print(deindex)
        somlist = []
