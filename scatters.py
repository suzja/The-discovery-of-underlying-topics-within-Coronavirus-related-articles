import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import csv
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm, preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import collections
from sklearn.cluster import KMeans
import re

def dummy(doc):
    return doc

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


if __name__ == '__main__':
    stop_words = stopwords.words("dutch")
    count1=0;
    count2=0;
    data = pd.read_csv('csvbestanden/samendata.csv', sep="|", error_bad_lines=False)


    data[' Text'] = data[' Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


    data['text_processed'] = \
    data[' Text'].map(lambda x: re.sub('[,\.!?]', '', x))

    data['text_processed'] = \
    data['text_processed'].map(lambda x: x.lower())
    data['text_processed'] = data['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


    dedata = data.text_processed.values.tolist()
    dedata_words = list(sent_to_words(dedata))



    # source = data['Title'].tolist()
    # source2 = data['Source'].tolist()
    # text = data[' Text'].tolist()
    counter = collections.Counter(dedata)
    # print(counter)

    y = data['text_processed'].astype('category').cat.codes
    # y = data[' Text'].astype('category').cat.codes
    # print(data['Source'])
    vectorizer = TfidfVectorizer(max_features=5000)
    # X = vectorizer.fit(data)

    X = vectorizer.fit_transform(data.values.tolist())
    true_k = 3
    clf = KMeans(n_clusters = true_k, init = 'k-means++', max_iter = 5000, n_init = 10)
    # clf = svm.SVC(max_iter=5000, cache_size=50000)
    kmeans = clf.fit(X, y)
    labels = kmeans.predict(X)



    model = TSNE(random_state=2)
    tsne_features = model.fit_transform(X)



    df = pd.DataFrame()
    df['x'] = tsne_features[:,0]
    df['y'] = tsne_features[:,1]

    centroids = np.array(kmeans.cluster_centers_)
    cent = pd.DataFrame()
    cent['X'] = centroids[:,0]
    cent['Y'] = centroids[:,1]
    # print(df)

    # print(centroids)
    sns.scatterplot(x='x', y='y', data=df, palette=['red', 'blue', 'green'])
    sns.scatterplot(x = 'X', y= 'Y', data = cent, marker = "x", color = 'r')
    plt.legend(loc = 'best')
    plt.show()
