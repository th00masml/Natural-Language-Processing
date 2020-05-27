# Simple preprocessing of url with pad sequences
from pprint import pprint
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pprint import pprint

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from summarizer import Summarizer

# Download the data from the website
def get_only_text(url):
    """
    Return title and text of article defined
    as url
    """
    page = urlopen(url)
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))

    print("=====================")
    print(text)
    print("=====================")

    return soup.title.text, text

url = 'https://www.gjclokhorst.nl/putnam.html'
text = get_only_text(url)
text = str(text)

body = text

"""model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
pprint(full)"""

 # Customized summarizer
from transformers import *

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)

model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

result = model(body, min_length=60)
full = ''.join(result)
print(full)
