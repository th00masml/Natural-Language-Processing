# Supervised Classification
# Gender Identification
#  Names ending in a, e and i are likely to be female,
#  while names ending in k, o, r, s and t are likely to be male
import nltk
from sklearn.preprocessing import LabelEncoder
from pprint import pprint

def gender_features(word):
     return {'last_letter': word[-1]}
print(gender_features('Ludwig'))

# Create dataset
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
 [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)

# Train classifier
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
labels = [(label) for label in featuresets]

# Evaluate classifier
print(classifier.classify(gender_features('John')))
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))

# Chose features
def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

print(gender_features2('Bartek'))

# Train Naive Bayes Model
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

# Evaluate model
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

# Split the data and evaluate it once again
train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]

pprint(train_set[:10])
print(type(train_set))

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))

"""
# Try KNN
from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier(n_neighbors=3)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train_encoded=le.fit_transform(labels)
print(train_encoded)
"""





