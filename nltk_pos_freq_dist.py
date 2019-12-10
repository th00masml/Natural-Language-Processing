import nltk
from nltk.corpus import indian

indian_pos = indian.tagged_words()

# Somehow tagset='universal' is not working. Need to check it
tag_fd = nltk.FreqDist(tag for (word, tag) in indian_pos)
print(tag_fd.most_common()[:5])

tag_fd.plot(cumulative=True)
