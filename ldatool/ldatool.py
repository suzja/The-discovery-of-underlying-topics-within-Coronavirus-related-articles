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


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    stop_words = stopwords.words("dutch")
    # print(stop_words)
    articles = pd.read_csv('csvbestanden/nudata.csv', sep="|", error_bad_lines=False)

    articles['Text'] = articles['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


    articles['text_processed'] = \
    articles['Text'].map(lambda x: re.sub('[,\.!?]', '', x))

    articles['text_processed'] = \
    articles['text_processed'].map(lambda x: x.lower())
    articles['text_processed'] = articles['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


    dedata = articles.text_processed.values.tolist()
    dedata_words = list(sent_to_words(dedata))
    dedata_words.map(lambda x: re.sub('[,\.!?]', '', x))
    nlp = spacy.load("nl_core_news_sm")
    excluded_tags = { "VERB", "ADV", "ADP", "PROPN"}
    poswords = []
    for i in dedata_words:
        nlpdata = nlp(str(i))
        for j in nlpdata:
            if (j.pos_ not in excluded_tags or is_punct(j) == False):
                poswords.append(j)
                # print(j)
    print(poswords[:10])
    # print(dedata_words)
    # [re.sub(r'\W', '', i) for i in x]

    # dattt = [re.sub("[^a-zA-Z]", " ", i) for i in dedata_words]
    # for i in dedata_words:

    #
    # print(dedata_words[:1][0][:30])
    #
    # dictword = corpora.Dictionary(dedata_words)
    # texts = dedata_words
    # corpus = [dictword.doc2bow(text) for text in texts]
    # print(corpus[:1][0][:30])
    #
    # num_topics = 10
    # lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictword, num_topics=num_topics)
    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
    #
    #
    # # Visualize the topics
    # # pyLDAvis.enable_notebook()
    # LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(num_topics))
    # # # this is a bit time consuming - make the if statement True
    # # # if you want to execute visualization prep yourself
    # if 1 == 1:
    #     LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictword)
    #     with open(LDAvis_data_filepath, 'wb') as f:
    #         pickle.dump(LDAvis_prepared, f)# load the pre-prepared pyLDAvis data from disk
    #
    # with open(LDAvis_data_filepath, 'rb') as f:
    #     LDAvis_prepared = pickle.load(f)
    #
    # pyLDAvis.save_html(LDAvis_prepared, './ldavis_nu'+ str(num_topics) +'.html')
    #
    # LDAvis_prepared
    #






# print(articles['text_processed'].head())

# # Join the different processed titles together.
# long_string = ','.join(list(articles['text_processed'].values))# Create a WordCloud object
# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
# wordcloud.generate(long_string)# Visualize the word cloud
# wordcloud.to_image()
# # plt.show()
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
