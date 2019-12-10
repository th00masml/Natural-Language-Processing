import nltk
from nltk.corpus import brown
from pprint import pprint

# Prints words that follows certain word, like time
brown_learned_text = brown.words(categories='learned')
print(sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'time'))) # Interesting line

# Do the same 'using tagged'_words method
brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'time']
fd = nltk.FreqDist(tags)

# Using '.tabulate' method. Check comparison with pprint
print(fd.tabulate())
pprint(fd)

# Bigger context
def process(sentence):
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1, w2, w3)

for tagged_sent in brown.tagged_sents():
     process(tagged_sent)

# Most ambiguous words related to their pos tag
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag)
                                for (word, tag) in brown_news_tagged)

for word in sorted(data.conditions()):
     if len(data[word]) > 3:
         tags = [tag for (tag, _) in data[word].most_common()]
         print(word, ' '.join(tags))

