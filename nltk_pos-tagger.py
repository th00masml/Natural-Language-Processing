import nltk
from nltk import word_tokenize
from pprint import pprint

raw = """We can defne a neural network that can learn to recognize objects",
             in less than 100 lines of code. However, after training, it is characterized,
             by millions of weights that contain the knowledge about many object types,
             across visual scenes. Such networks are thus dramatically easier to understand,
             in terms of the code that makes them than the resulting properties, such,
             as tuning or connections. In analogy, we conjecture that rules for development,
             and learning in brains may be far easier to understand than their resulting,
             properties. The analogy suggests that neuroscience would beneft from a focus,
             on learning and development."""

text = word_tokenize(raw)
pos_tagger = nltk.pos_tag(text)

print(pos_tagger)
pprint(pos_tagger)

# To lowercase
text = raw.lower()
print(text)

text = nltk.Text(text)

print(text.similar('network'))
print(text)