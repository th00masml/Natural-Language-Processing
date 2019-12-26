import nltk
from nltk.corpus import brown
from pprint import pprint
import pylab

brown_tagged_sents = brown.tagged_sents(categories='religion')
brown_sents = brown.sents(categories='religion')

# Create default tagger
tags = [tag for (word, tag) in brown.tagged_words(categories='religion')]
print(nltk.FreqDist(tags).max())

raw = 'The more I think about language, the more it amazes me that people ever understand each other at all'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
pprint(default_tagger.tag(tokens))

# Evaluate performance
print(default_tagger.evaluate(brown_tagged_sents))

# Regular Expression Tagger
patterns = [
     (r'.*ing$', 'VBG'),                # gerunds
     (r'.*ed$', 'VBD'),                 # simple past
     (r'.*es$', 'VBZ'),                 # 3rd singular present
     (r'.*ould$', 'MD'),                # modals
     (r'.*\'s$', 'NN$'),                # possessive nouns
     (r'.*s$', 'NNS'),                  # plural nouns
     (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                      # nouns (default)
 ]

print(patterns)
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[1])

# Evaluate regex tagger
print(regexp_tagger.evaluate(brown_tagged_sents))

# Try regex tagger on smaller sample
raw = 'If a machine is expected to be infallible, it cannot also be intelligent'
tokens_turing = nltk.word_tokenize(raw)
regexp_tagger_turing = nltk.RegexpTagger(patterns)
print(regexp_tagger_turing.tag(tokens_turing))

# Lookup Tagger
fd = nltk.FreqDist(brown.words(categories='religion'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='religion'))
most_freq_words = fd.most_common(100)

# Most likely tags for most freq words
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print(baseline_tagger.evaluate(brown_tagged_sents))


sent = brown.sents(categories='religion')[1]
print(baseline_tagger.tag(sent))

# Set default tagger to 'NN'
baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                      backoff=nltk.DefaultTagger('NN'))

# Create and evaluate lookup taggers with range of sizes
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='religion'))

# Plot freq dist using pylab
def display():
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

print(display())

# N-Gram Tagging
# Unigram tagging
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='religion')
brown_sents = brown.sents(categories='religion')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[100]))

# Train tagger
# Split data
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Train and evaluate
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

# N-Gram Tagging
bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.tag(brown_sents[99]))

test_sent = brown_sents[100]
print(bigram_tagger.tag(test_sent))

# Very poor performance on new data
# Sparse data problem. The bigger n the more complex context
# Related to the precision/recall trade-off in information retrieval
# NLTK taggers are designed to work with lists of sentences,
# where each sentence is a list of words
print(bigram_tagger.evaluate(test_sents))

# Combined Taggers
# General procedure as fallows:
# 1. Try tagging the token with the bigram tagger.
# 2. If the bigram tagger is unable to find a tag for the token, try the unigram tagger.
# 3. If the unigram tagger is also unable to find a tag, use a default tagger.
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print(t2.evaluate(test_sents))

# Store tagger
from pickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

# Load saved tagger
from pickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()

# Using saved tagger
text = """The idea behind digital computers may be explained by saying that 
these machines are intended to carry out any operations which could be done by a human computer."""
tokens = text.split()
print(tagger.tag(tokens))

# Tagging errors with confusion matrix
# It charts expected tags (the gold standard) against actual tags generated by a tagger
test_tags = [tag for sent in brown.sents(categories='religion')
                  for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='religion')]
print(nltk.ConfusionMatrix(gold_tags, test_tags))

# Transformation-Based Tagging
# General idea: guess the tag of each word, then go back and fix the mistakes
from nltk.tbl import demo as brill_demo
print(brill_demo.demo())
