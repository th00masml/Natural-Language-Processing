from gensim import corpora
from gensim import models
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

documents = ["We can defne a neural network that can learn to recognize objects",
             "in less than 100 lines of code. However, after training, it is characterized",
             "by millions of weights that contain the knowledge about many object types",
             "across visual scenes. Such networks are thus dramatically easier to understand",
             "in terms of the code that makes them than the resulting properties, such",
             "as tuning or connections. In analogy, we conjecture that rules for development",
             "and learning in brains may be far easier to understand than their resulting",
             "properties. The analogy suggests that neuroscience would beneft from a focus",
             "on learning and development."]

# Create the Dictionary and Corpus
mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]

# Show the Word Weights in Corpus
for doc in corpus:
    print([[mydict[id], freq] for id, freq in doc])

# Create the TF-IDF model
tfidf = models.TfidfModel(corpus, smartirs='ntc')

# Show the TF-IDF weights
for doc in tfidf[corpus]:
    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])

for doc in tfidf[corpus]:
    print([[mydict[id], freq] for id, freq in doc])


