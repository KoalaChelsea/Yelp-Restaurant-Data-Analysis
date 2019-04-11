#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:24:13 2018

@author: Shaoyu
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import gc
from wordcloud import WordCloud
import squarify
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re 
import gensim 
from gensim import corpora
nltk.download('wordnet')
#%%

# =============================================================================
# Review Data Exploratory Analysis
# =============================================================================

yelp = pd.read_csv('Rawdata/yelp.csv')
yelp.head(10)
yelp['text length'] = yelp['text'].apply(len)
yelp['num_uniq_words'] = yelp['text'].apply(lambda x: len(set(str(x).split())))
yelp['num_chars'] = yelp['text'].apply(lambda x: len(str(x)))
yelp['num_stopwords'] = yelp['text'].apply(lambda x: 
    len([w for w in str(x).lower().split() if w in set(stopwords.words('english'))]))
    
    
# Distribution of text feature
f, ax = plt.subplots(2,2, figsize = (14,10))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.distplot(yelp['text length'],bins=100,color='r', ax=ax1)
ax1.set_title('Distribution of Number of chars')

sns.distplot(yelp['num_uniq_words'],bins=100,color='b', ax=ax2)
ax2.set_title('Distribution of Unique words')

sns.distplot(yelp['num_chars'],bins=100,color='y', ax=ax3)
ax3.set_title('Distribution of Char words')

sns.distplot(yelp['num_stopwords'],bins=100,color='r', ax=ax4)
ax4.set_title('Distribution of Stop words')

g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)


# =============================================================================
# Review Data Exploratory Analysis
# =============================================================================


yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
X = yelp_class['text']
y = yelp_class['stars']

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)   
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
X = bow_transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap="YlGnBu",\
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])


# =============================================================================
# Topic Modeling
# =============================================================================

cloud = WordCloud(width=1000, height= 800,max_words= 300).generate(' '.join(yelp['text'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off');

# clean text
lemma = WordNetLemmatizer()

def clean_text(doc):
    corpus = []
    for c in range(0, doc.shape[0]):
        stop_free = ' '.join([i for i in doc['text'][c].lower().split() if i not in set(stopwords.words('english'))])
        puct_free = ''.join(i for i in stop_free if i not in set(string.punctuation))
        normalized = [lemma.lemmatize(word) for word in puct_free.split()]
        corpus.append(normalized)
    return corpus
doc_tips = clean_text(yelp)

dictionary = corpora.Dictionary(doc_tips)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_tips]


ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics= 5, id2word= dictionary, passes=20)

for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words= 5):
    print('Topic {}: words'.format(topic[0]))
    topic_word = [w for (w,val) in topic[1]]
    print(topic_word)

# Top topics in document
tp = ldamodel.top_topics(doc_term_matrix,topn=20,dictionary=dictionary)

# tuple unpacking
label = [] 
value = []

f,ax = plt.subplots(1,2,figsize = (14,6))
ax1.set_title(tp[0])
ax1,ax2 = ax.flatten()
for i,k in tp[0][0]:
    label.append(i)
    value.append(k)
sns.barplot(label,value,palette='BrBG', ax=ax1)

label = [] 
value = []
for i,k in tp[1][0]:
    label.append(i)
    value.append(k)
sns.barplot(label,value,palette='RdBu_r', ax= ax2);



