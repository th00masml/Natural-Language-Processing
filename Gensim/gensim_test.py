# Define decuments
documents = ["Death is not an event in life: we do not live to experience death",
    "If we take eternity to mean not infinite temporal duration but timelessness",
    "then eternal life belongs to those who live in the present",
    "Our life has no end in the way in which our visual field has no limits"]

print(documents)

""" Prepare the text """
from pprint import pprint  # pretty-printer
from collections import defaultdict

# Remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
for document in documents] # Split all the documents that are not in stoplist
print(texts)

# Remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
for text in texts]
pprint(texts)

""" Token to id """
from gensim import corpora
dictionary = corpora.Dictionary(texts)
dictionary.save('C:\\Users\\bnawa\\Data\\deerwester.dict')  # store the dictionary, for future reference
print(dictionary)
print(dictionary.token2id)

""" Tokens to vectors"""
new_doc = "Death is eternity"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

""" Serialize the corpus """
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('C:\\Users\\bnawa\\Data\\deerwester.mm', corpus)  # store to disk, for later use
print(corpus)
print (help(corpus))


