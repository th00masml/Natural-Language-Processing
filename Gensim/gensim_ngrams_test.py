from gensim import corpora
from gensim.models import Phrases
from pprint import pprint
from gensim import corpora
from gensim.utils import simple_preprocess
import gensim
from pprint import pprint

documents = ["We can define a neural network that can learn to recognize objects",
             "in less than 100 lines of code. However, after training, it is characterized",
             "by millions of weights that contain the knowledge about many object types",
             "across visual scenes. Such networks are thus dramatically easier to understand",
             "in terms of the code that makes them than the resulting properties, such",
             "as tuning or connections. In analogy, we conjecture that rules for development",
             "and learning in brains may be far easier to understand than their resulting",
             "properties. The analogy suggests that neuroscience would beneft from a focus",
             "on learning and development."]

tokenized_list = [simple_preprocess(doc) for doc in documents]

mydict = corpora.Dictionary()

mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
bigram = gensim.models.phrases.Phrases(documents, min_count=3, threshold=10)

pprint(bigram[tokenized_list[0]])