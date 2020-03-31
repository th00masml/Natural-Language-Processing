#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import nltk
#from nltk.corpus import stopwords
#set(stopwords.words('english'))

stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
        'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
        'won', "won't", 'wouldn', "wouldn't"]

data = pd.read_csv('path')
print(data.head(5))


# Already preprocessed
data['Target Column Preprocessed'] = data['Target Column '].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Target Column Preprocessed'] = data['Target Column Preprocessed'].str.replace('[^\w\s]','')
data['Target Column Preprocessed'] = data['Target Column Preprocessed'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(data.head(5))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(data['Target Column Preprocessed'].values)
tfidf.shape

from sklearn.cluster import KMeans
model_tf = KMeans(n_clusters = 5, n_jobs = -1,random_state=99)
model_tf.fit(tfidf)

labels_tf = model_tf.labels_
cluster_center_tf=model_tf.cluster_centers_



terms1 = tfidf_vect.get_feature_names()
from sklearn import metrics
silhouette_score_tf = metrics.silhouette_score(tfidf, labels_tf, metric='euclidean')
print(silhouette_score_tf)

df = data
df['Tfidf KMeans Clus Label'] = model_tf.labels_
print(df.head(5))


print("Top terms per cluster:")
order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]
for i in range(5):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms1[ind], end='')
        print()



# Plot distribution of clusters
import matplotlib.pyplot as plt
plt.bar([x for x in range(5)], df.groupby(['Tfidf KMeans Clus Label'])['Target Column '].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()


# Reading a review which belong to each group.
for i in range(5):
    print("4 review of assigned to cluster ", i)
    print("-" * 70)
    print(df.iloc[df.groupby(['Tfidf KMeans Clus Label']).groups[i][5]]['Target Column '])
    print('\n')
    print(df.iloc[df.groupby(['Tfidf KMeans Clus Label']).groups[i][10]]['Target Column '])
    print('\n')
    print(df.iloc[df.groupby(['Tfidf KMeans Clus Label']).groups[i][20]]['Target Column '])
    print('\n')
    print("_" * 70)


df.to_csv('path')






