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
import json
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import numpy as np
import os
import spacy
from spacy.lang.nl.examples import sentences
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
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


    media = ["nudata", "nrcdata", "standaarddata", "volkskrantdata"]
    media1 = ["standaarddata"]

    for i in range(len(media)):

        print(media[i])

        wordlist = make_lda(media[i])
        poswords = list(sent_to_words(wordlist))
        woordenlos = []

        nlp = spacy.load("nl_core_news_sm")
        excluded_tags = { "VERB", "ADP", "PUNCT", "NUM", "SYM", "AUX", "ADV", "CONJ", "DET", "PART", "PRON", "SCONJ", "X"}

        for q in range(len(poswords)):
            nlpdata = nlp(str(q))
            for z in nlpdata:
                if (z.pos_ not in excluded_tags):
                    woordenlos.append(z)

        dictword = corpora.Dictionary(woordenlos)
        texts = woordenlos
        corpus = [dictword.doc2bow(text) for text in texts]

        tfidf_vectorizer=TfidfVectorizer(use_idf=True)
        tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(wordlist)
        # vectorizer.get_feature_names_out()

        tfidf = tfidf_vectorizer_vectors.todense()
        # TFIDF of words not in the doc will be 0, so replace them with nan
        tfidf[tfidf == 0] = np.nan
        # Use nanmean of numpy which will ignore nan while calculating the mean
        means = np.nanmean(tfidf, axis=0)
        # convert it into a dictionary for later lookup
        means = dict(zip(tfidf_vectorizer.get_feature_names(), means.tolist()[0]))
        # print(means)
        tfidf = tfidf_vectorizer_vectors.todense()
        # Argsort the full TFIDF dense vector
        # print(tfidf[:1][0][:30])
        ordered = np.argsort(tfidf*-1)
        words = tfidf_vectorizer.get_feature_names()

        top_k = 40
        # gemlijst = []
        result = { }

        result.clear()
        # print(result)
        lijst = []
        grote =[]
        stop_words = stopwords.words("dutch")
        stop_words.extend(['nrc', 'nrc.nl','nu', 'kun', 'nunl', 'ad', 'twitter', 'nu.nl', 'NU.nl', 'https', 'telegraaf', 'telegraaf.nl', 'the', 'to', 'volkskrant', 'volkskrant.nl', 'dagelijksestandaard', 'we'])

        for k, doc in enumerate(wordlist):
            # print(doc)
            # Pick top_k from each argsorted matrix for each doc
            for t in range(top_k):
                # Pick the top k word, find its average tfidf from the
                # precomputed dictionary using nanmean and save it to later use
                result[words[ordered[k,t]]] = means[words[ordered[k,t]]]

            grote.append(list(result))
            # grote.append(";")
            result.clear()
            # print(lijst)
            # lijst = []
        nieuwe = []
        tijdelijke = []

        for p in range(len(grote)):
            tokens_without_sw = [word for word in grote[p] if not word in stop_words]
            tokens_without_sw = [word for word in grote[p] if not word in excluded_tags]
            # print (tokens_without_sw)
            for j in range(len(tokens_without_sw)):
                nieuwe.append(tokens_without_sw[j])


        with open('woordentext' + str(media[i]) + '.txt', 'w') as f:
            f.write(str(nieuwe))
        # print(nieuwe)
