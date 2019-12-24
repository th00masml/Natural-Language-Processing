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