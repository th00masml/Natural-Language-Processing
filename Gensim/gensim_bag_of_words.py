from gensim import corpora
from gensim.utils import simple_preprocess
from pprint import pprint

# Create list
my_docs = ["Tautologies and contradictions are not, however, nonsensical.",
           "They are part of the symbolism, much as '0' is part of the symbolism",
           "of arithmetic."]

# Tokenize
tokenized_list = [simple_preprocess(doc) for doc in my_docs]

# Create the Corpus
mydict = corpora.Dictionary()

# Create bag of words using doc2bow function
mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
pprint(mycorpus)

# Convert id's to words
word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]
pprint(word_counts)

# Create bag of words from text file
# Using __iter__ method
from gensim.utils import simple_preprocess
from smart_open import smart_open
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

class BoWCorpus(object):
    def __init__(self, path, dictionary):
        self.filepath = path
        self.dictionary = dictionary

    def __iter__(self):
        global mydict
        for line in smart_open(self.filepath, encoding='latin'):
            # tokenize
            tokenized_list = simple_preprocess(line, deacc=True)

            # create bag of words
            bow = self.dictionary.doc2bow(tokenized_list, allow_update=True)

            # update the source dictionary (OPTIONAL)
            mydict.merge_with(self.dictionary)

            # lazy return the BoW
            yield bow


# Create the Dictionary
mydict = corpora.Dictionary()

# Create the Corpus
bow_corpus = BoWCorpus('C:\\Users\\bnawa\\Data\\tractatus.txt', dictionary=mydict)  # memory friendly

# Print the token_id and count for each line.
for line in bow_corpus:
    print(line)