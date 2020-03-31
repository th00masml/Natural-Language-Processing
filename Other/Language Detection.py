#!/usr/bin/env python
# coding: utf-8

from langdetect import detect
from langdetect import detect_langs
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('path')
print(data.head(5))


data['Target Column'] = data['Target Column'].apply(detect)
print(data.head(5))

print(data['Target Column'].hist())
