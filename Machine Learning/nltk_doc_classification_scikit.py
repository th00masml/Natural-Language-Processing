# Using movie reviews dataset
# Dataset contains 1000 positive and 1000 negative processed reviews

import random
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
print(documents[:10])
