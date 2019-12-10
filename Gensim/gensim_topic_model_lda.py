# Import packages and stopwords
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
import re
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
stop_words = stopwords.words('english')
stop_words = stop_words + ['com', 'edu', 'subject', 'lines', 'organization', 'would', 'article', 'could']

# Import dataset
dataset = api.load("fake-news")
data = [d for d in dataset]
pprint(data[:1])

# Prepare the data (remove stopwords and lemmatize)
data_processed = []

for i, doc in enumerate(data[:100]):
    doc_out = []
    for wd in doc:
        if wd not in stop_words:  # remove stopwords
            lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
            if lemmatized_word:
                doc_out = doc_out + [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
        else:
            continue
    data_processed.append(doc_out)

# Print a small sample
print(data_processed[0][:5])
