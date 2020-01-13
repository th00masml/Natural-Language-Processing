# Find most common suffixes
from nltk.corpus import brown
import nltk
from pprint import pprint

suffix_fdist = nltk.FreqDist()
for word in brown.words():
     word = word.lower()
     suffix_fdist[word[-2:]] += 1
     suffix_fdist[word[-3:]] += 1
     suffix_fdist[word[-1:]] += 1

# Top 100 most common suffixes
common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
pprint(common_suffixes)

# Create feature extraction function with common suffixes
def pos_features(word):
     features = {}
     for suffix in common_suffixes:
         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
     return features

# Create Decision Tree Classifier to extract pos
tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]

classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

print(classifier.classify(pos_features('God')))

print(classifier.pseudocode(depth=4))