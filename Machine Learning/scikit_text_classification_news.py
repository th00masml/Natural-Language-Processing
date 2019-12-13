# Classification on 20 Newsgroups dataset

"""

The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents,
partitioned (nearly) evenly across 20 different newsgroups.
To the best of my knowledge, it was originally collected by Ken Lang,
probably for his Newsweeder: Learning to filter netnews paper, though he does not explicitly mention this collection.
The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques,
such as text classification and text clustering.

"""
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
pprint(list(twenty_train.target_names))

# Using Count Vectorizer to extract features from data
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

print(X_train_counts)
print(X_train_counts.shape)

# Using TF-IDF to reduce data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf)
print(X_train_tfidf.shape)

# Using Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) # Training vectors, Target values

# Build pipeline to chain multiple estimators, defined before
# This is not added value, we can do all the previous steps this way
# using much less code
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
 ])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

print(text_clf)

# Test NB Classification model
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
# Compare predicted values with target values using simple numpy comparasion
print(np.mean(predicted == twenty_test.target))

# Build pipeline with different algorithm (Support Vector Machine (SVM))
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted_svm = text_clf_svm.predict(twenty_test.data)

# Test model
print(np.mean(predicted_svm == twenty_test.target))

# Using Grid Search CV to find correct hyper parameters
# Create list of parameters for which we would like to do performance tuning
# After name of classifier name we are giving list of parameters to chose which is optimal

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3)}

# Creates instance of classifier with optimal parameters
# Parameter 'n_jobs=-1' consumes all available resources as and when they become available
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

print(gs_clf.best_score_)
print(gs_clf.best_params_)

# Find parameters to SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)

# Adjust stemming to increase score
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                      ('tfidf', TfidfTransformer()),
                      ('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
print(np.mean(predicted_mnb_stemmed == twenty_test.target))
