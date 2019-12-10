import nltk
from nltk.corpus import indian

indian_pos = indian.tagged_words()
# Somehow tagset='universal' is not working. Need to check it
tag_fd = nltk.FreqDist(tag for (word, tag) in indian_pos)
#print(tag_fd.most_common()[:5])
#tag_fd.plot(cumulative=True)

""" Nouns """

import nltk

word_tag_pairs = nltk.bigrams(indian_pos)
# Prints the type of word before noun
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NN']
fdist = nltk.FreqDist(noun_preceders)
print([tag for (tag, _) in fdist.most_common()])

""" Verbs """

# tagset='universal' works here. Issue connected with corpus
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
print([wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB'])

# Frequency-ordered list of tags given a word
cfd1 = nltk.ConditionalFreqDist(wsj)

print(cfd1['he'].most_common())

# Reverse previous operations. Show list of words, likely for given a tag
wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
print(list(cfd2['NN']))

# Finds most freq nouns of each noun pos type
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())

tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
     print(tag, tagdict[tag])