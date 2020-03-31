#!/usr/bin/env python
# coding: utf-8

import gensim
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
data[Target Column Preprocessed] = data[Target Column].apply(lambda x: " ".join(x.lower() for x in x.split()))
data[Target Column Preprocessed] = data[Target Column Preprocessed].str.replace('[^\w\s]','')
data[Target Column Preprocessed] = data[Target Column Preprocessed].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(data.head(5))


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
bow = count_vect.fit_transform(data[Target Column Preprocessed].values)
bow.shape


from sklearn.cluster import KMeans
model = KMeans(n_clusters=5,init='k-means++', n_jobs = -1,random_state=99)
model.fit(bow)

labels = model.labels_
cluster_center=model.cluster_centers_


from sklearn import metrics
silhouette_score = metrics.silhouette_score(bow, labels, metric='euclidean')
print(silhouette_score)

df = data
df['Bow KMeans Clus Label'] = model.labels_ # the last column you can see the label numebers
print(df.head(5))

# Plot distribution of clusters
import matplotlib.pyplot as plt
plt.bar([x for x in range(5)], df.groupby(df['Bow KMeans Clus Label'])[Target Column].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = count_vect.get_feature_names()
for i in range(5):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :8]:
        print(' %s' % terms[ind], end='')
        print()


for i in range(10):
    print("A review of assigned to cluster ", i)
    print("-" * 70)
    print(df.iloc[df.groupby(['Bow KMeans Clus Label']).groups[i][0]][Target Column])
    print('\n')
    print("_" * 70)


# 3 random reviews for cluster 5

print(df.iloc[df.groupby(['Bow KMeans Clus Label']).groups[0][3]][Target Column])
print("_" * 70)
print(df.iloc[df.groupby(['Bow KMeans Clus Label']).groups[0][15]][Target Column])
print("_" * 70)
print(df.iloc[df.groupby(['Bow KMeans Clus Label']).groups[0][2]][Target Column])



df.to_csv('path')





