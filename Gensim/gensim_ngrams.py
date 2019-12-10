import gensim
from gensim import corpora
import gensim.downloader as api
from pprint import pprint

dataset = api.load("fake-news")
dataset = [wd for wd in dataset]

dct = corpora.Dictionary(dataset)
corpus = [dct.doc2bow(line) for line in dataset]

# Construct bigrams model
bigram = gensim.models.phrases.Phrases(dataset, min_count=3, threshold=10)

# Construct bigrams
pprint(bigram[dataset[0]])

# Construct trigrams model
trigram = gensim.models.phrases.Phrases(bigram[dataset], threshold=10)

# Construct trigram
pprint(trigram[bigram[dataset[0]]])