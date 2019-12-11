"""Count Vectoring"""
from sklearn.feature_extraction.text import CountVectorizer

text = raw = ['We can define a neural network that can learn to recognize objects, in less than 100 lines of code.']

# Define vectorizer and fit it
vectorizer = CountVectorizer()
vectorizer.fit(text)

# Check the output of Count Vectorizer
print(vectorizer.vocabulary_)

# Transform text to vector
vector = vectorizer.transform(text)
print(vector.shape)
print(type(vector))
print(vector.toarray())

# As we can see here, only word 'can' occur two times
print(vector)

"""TF-IDF"""
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tdidf = TfidfVectorizer()
vectorizer_tdidf.fit(text)

print(vectorizer_tdidf.vocabulary_)
print(vectorizer_tdidf.idf_)

vector_tdidf = vectorizer_tdidf.transform([text[0]])

print(vector_tdidf.shape)
print(vector_tdidf.toarray())

"""Hashing Vectoring"""
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer_hash = HashingVectorizer(n_features=20)
vector_hash = vectorizer_hash.transform(text)

print(vector_hash.shape)
print(vector_hash.toarray())