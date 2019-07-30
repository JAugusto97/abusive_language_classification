#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from gensim.models.phrases import Phrases, Phraser
import ast


# In[2]:
print('Reading data...\n')
# Reading data and converting string of list to list
data = pd.read_csv('data/corpus/twitter_text.csv', header=None, names=['text'])
data = data['text'].apply(lambda x: ast.literal_eval(x))


# In[3]:

print('Generating bigrams...\n')
# Generate bigrams
phrases = Phrases(data, min_count=20, delimiter=b' ')
bigram = Phraser(phrases)
sentences = bigram[data]


# In[4]:


sequential_text = []
for sentence in sentences:
    for token in sentence:
        sequential_text.append(token)

frequent_terms = FreqDist(sequential_text)


# In[5]:


print('top frequent words:')
for word, count in frequent_terms.most_common()[:10]:
    print(word,count)


# In[ ]:


print('total amount of sentences:', len(sentences))


# In[6]:


print('total amount of tokens:', len(sequential_text))


# In[7]:


# number of unique tokens
print('amount of unique tokens:', len(set(sequential_text)), '\n')


# In[8]:


vector_size = input('vector size:')
window_size = input('window size:')


# In[9]:


print('Generating cbow with vector size='+vector_size+' and window size='+window_size)
cbow = Word2Vec(sentences, size=int(vector_size), window=int(window_size), sg=0)
print('Generating skipgram with vector size='+vector_size+' and window size='+window_size)
skipgram = Word2Vec(sentences, size=int(vector_size), window=int(window_size), sg=1)


# In[10]:


cbow.save('data/word_embeddings/twitter_cbow_'+str(vector_size)+'_'+str(window_size))
skipgram.save('data/word_embeddings/twitter_skipgram_'+str(vector_size)+'_'+str(window_size))

