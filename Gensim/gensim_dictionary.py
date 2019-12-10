import gensim
from gensim import corpora
from pprint import pprint

documents = ["We can defne a neural network that can learn to recognize objects",
             "in less than 100 lines of code. However, after training, it is characterized",
             "by millions of weights that contain the knowledge about many object types",
             "across visual scenes. Such networks are thus dramatically easier to understand",
             "in terms of the code that makes them than the resulting properties, such",
             "as tuning or connections. In analogy, we conjecture that rules for development",
             "and learning in brains may be far easier to understand than their resulting",
             "properties. The analogy suggests that neuroscience would beneft from a focus",
             "on learning and development."]

documents_2 = ["Humans are not neural networks. And yet, the brain has ubiquitous plasticity.",
               "Speciacally, we know that plasticity allows changes in the brain to enable",
               "good performance across new tasks. As such, it is hard to see how",
               "the arguments we made above about articial neural networks would not carry",
               "over to the human brain."]

# Tokenize the sentences
texts = [[text for text in doc.split()] for doc in documents]

# Create dictionary

dictionary = corpora.Dictionary(texts)
print(dictionary)
pprint(dictionary.token2id)

# Add second document
texts_2 = [[text for text in doc.split()] for doc in documents_2]
dictionary.add_documents(texts_2)

print(dictionary)
print(dictionary.token2id)

# Create dictionary from text file

from gensim.utils import simple_preprocess
from smart_open import smart_open
import os

dictionary_text = corpora.Dictionary(simple_preprocess(line, deacc=True)
                                for line in open('C:\\Users\\bnawa\\Data\\tractatus.txt', encoding='utf-8'))
print(dictionary_text)
print(dictionary_text.token2id)

# Save the Dictionary and Corpus
dictionary.save('wittgenstein_quotes.dict')
corpora.MmCorpus.serialize('bow_wittgenstein_quotes.mm', bow_wittgenstein_quote)

# Load saved files
loaded_dict = corpora.Dictionary.load('wittgenstein_quotes.dict')

corpus = corpora.Dictionary('bow_wittgenstein_quotes.mm')
for line in corpus:
    print(line)

