import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
from pprint import pprint

# Set up algorithm and tokenizer
sid = SentimentIntensityAnalyzer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Read the data
f = open(r'C:\Users\bnawa\turing_imitation-game.txt')
imitation_game1 = f.read()
print(type(imitation_game1))

# Tokenize to sentences
sentences = tokenizer.tokenize(imitation_game1)
pprint(sentences[:10])

# Tokenize to words
words = word_tokenize(imitation_game1)
fdist = nltk.FreqDist(words)
pprint(fdist.most_common(20))

# Create new dataset with sentences that contains word "machine"
import re
r = re.compile(".* machine .*")
machinelist = list(filter(r.match, sentences))
pprint(machinelist[:10])

# Print each sentence with scores
for sentence in machinelist:
    print(sentence)
    scores = sid.polarity_scores(sentence)
    for key in sorted(scores):
        print('{0}: {1}, '.format(key, scores[key]), end='')
