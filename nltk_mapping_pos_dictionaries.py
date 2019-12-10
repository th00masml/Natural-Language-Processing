from collections import defaultdict
import nltk
from pprint import pprint

# Create a default dictionary that maps each word to its replacement
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [word for (word, _) in vocab.most_common(1000)]
mapping = defaultdict(lambda: 'UNK')
for v in v1000:
     mapping[v] = v

alice2 = [mapping[v] for v in alice]
print(alice2[:10])

# Incrementally Updating a Dictionary
from collections import defaultdict

# Set up default dict to hold numbers of tags
counts = defaultdict(int)
from nltk.corpus import brown
for (word, tag) in brown.tagged_words(categories='religion', tagset='universal'):
     counts[tag] += 1
print(sorted(counts))

from operator import itemgetter
# Get tags with freq using, sorting by 'itemgetter'
# 'itemgeter' returns a function that can be called on some other sequence object to obtain the nth element
print(sorted(counts.items(), key=itemgetter(1), reverse=True))

# Gets list of tags
print([t for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)])

# Index words, according to their last two letters
last_letters = defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
     key = word[-2:] # Take last two letters
     last_letters[key].append(word) # And append it to word, in order to look for words with certain ending

print(last_letters['ed'])

# Use the same logic to create anagrams
anagrams = defaultdict(list)
for word in words:
     key = ''.join(sorted(word))
     anagrams[key].append(word)

print(anagrams['aeilnrt'])
