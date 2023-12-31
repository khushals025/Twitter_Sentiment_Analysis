# Twitter Sentiment Analysis

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1000/1*vp1M37AGMOFwCvLxVm62IA.jpeg" alt="Image Alt" width="1000">
</div>

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Data Pre-Processing](#data-pre-processing)


## 1. Introduction

- A significant amount of data is generated as well as being made available to internet users thanks to the development and growth of online technologies. The internet has developed into a forum for online education, idea sharing, and opinion exchange. Social networking services like Twitter, Facebook, and Google+ are quickly gaining popularity as a result of the ability for users to share and express their opinions on many subjects, engage in conversation with various communities, and broadcast messages globally.The study of sentiment in Twitter data has received a lot of attention.
- The major aim of this project is sentiment analysis of twitter data, which is useful for analyzing information in tweets when opinions are very unstructured, varied, and occasionally neutral. In this project, we present a comparative analysis, assessment metrics, and existing methods for opinion mining, such as lexicon-based methods and machine learning methods. We present research on twitter data streams using a variety of machine learning methods, including Bernoulli Naive Bayes, SVM (Support Vector Machine), Logistic Regression, and Neural Network.


## 2. Dependencies


```bash
import re
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import nltk
```

- ML Models
```bash

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Dense,GlobalMaxPooling1D,Embedding
from tensorflow.keras.models import Model

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```

- Sentiment Analysis
```bash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from wordcloud import WordCloud
```


## 3. Dataset

- This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment.
- dataset: <https://www.kaggle.com/datasets/kazanova/sentiment140>

## 4. Data Pre-processing

A tweet includes numerous perspectives about the data that are presented by various people in various ways. The Twitter dataset utilized in this study is already divided into two categories, negative and positive polarity, making it simple to perform a sentiment analysis on the data and see how different variables affect sentiment. Polarity in the raw data makes it particularly prone to redundancy and inconsistency.
Preprocessing of datasets and tweet include following points:

- Removing unused features from the dataset.
- Removing stopwords from tweets using NLTK. 
- Removing punctuations from the tweets.
- Removing repeating characters from the tweets. 
- Removing URL’s from the tweets.
- Removing numbers from the tweets.

### 4.1 Tokenization

- Process of breaking down the given text in natural language processing into the smallest unit in a sentence called a token. Punctuation marks, words, and numbers can be considered tokens.
- With the help of nltk.tokenize.SpaceTokenizer() method, we are able to extract the tokens from string of words on the basis of space between them by using tokenize.SpaceTokenizer() method.

```bash
from nltk.tokenize import SpaceTokenizer
tokenizer=SpaceTokenizer()
joinedtweets['text'] = joinedtweets['text'].apply(tokenizer.tokenize)
joinedtweets['text'].head()
```

<div align="center">
  <img src="https://thepythoncode.com/media/articles/tokenization-stemming-and-lemmatization-in-python/img1.png" alt="Image Alt" width="400">
</div>

### 4.2 Stemming 
- With stemming, words are reduced to their word stems. A word stem need not be the same root as a dictionary-based morphological root, it just is an equal to or smaller form of the word.

There are 3 types of stemming:

- Porter Stemmer
- Snowball Stemmer
- Lancaster Stemmer

We used Porter Stemmer in this Project. 

#### Porter-Stemmer Algorithm 

<div align="center">
  <img src="https://qph.cf2.quoracdn.net/main-qimg-187b045c480fa7c0b16869daa0661b5a" alt="Image Alt" width="600">
</div>

- The rules for replacing (or removing) a suffix will be given in the form as shown below.

(condition) S1 → S2

- This means that if a word ends with the suffix S1, and the stem before S1 satisfies the given condition, S1 is replaced by S2. The condition is usually given in terms of m in regard to the stem before S1.

(m > 1) EMENT →
Here S1 is ‘EMENT’ and S2 is null. This would map REPLACEMENT to REPLAC, since REPLAC is a word part for which m = 2.

<div align="center">
  <img src="https://vijinimallawaarachchi.files.wordpress.com/2017/05/ex.png" alt="Image Alt" width="400">
</div>

```bash
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
joinedtweets['text']= joinedtweets['text'].apply(lambda x: stemming_on_text(x))
joinedtweets['text'].head()
```


The conditions may contain the following:

- *S    –    the stem ends with S (and similarly for the other letters)
- *v*  –    the stem contains a vowel
- *d    –    the stem ends with a double consonant (e.g. -TT, -SS)
- *o    –    the stem ends cvc, where the second c is not W, X or Y (e.g. -WIL, -HOP)

- Note: These are just a few rules, in the actual algorithm there are many.

### 4.3 Lemmatization

- The aim of lemmatization, like stemming, is to reduce inflectional forms to a common base form. As opposed to stemming, lemmatization does not simply chop off inflections. Instead, it uses lexical knowledge bases to get the correct base forms of words.

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1128/format:webp/1*HLQgkMt5-g5WO5VpNuTl_g.jpeg" alt="Image Alt" width="700">
</div>

```bash
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import WordNetLemmatizer

lm = WordNetLemmatizer()  
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
joinedtweets['text'] = joinedtweets['text'].apply(lambda x: lemmatizer_on_text(x))
joinedtweets['text'].head()
```

### Positive word cloud

<div align="center">
  <img src="https://github.com/khushals025/Twitter_Sentiment_Analysis/blob/main/Positive.png?raw=true" alt="Image Alt" width="1000">
</div>


### Negative Word cloud


<div align="center">
  <img src="https://github.com/khushals025/Twitter_Sentiment_Analysis/blob/main/Negative.png?raw=true" alt="Image Alt" width="1000">
</div>

### 4.4 TF-IDF Vectorization 

- Term Frequency (tf)- It gives us the recurrence of the word in each report in the corpus. It is the proportion of the number of times the word shows up in a report contrasted with the all-out the number of words in that record. It increments as the quantity of events of that word inside the record increments.

- Inverse Data Frequency (idf)- It is used to figure out the heaviness of uncommon words over all reports in the corpus. The words that occur seldom in the corpus have a high IDF score.

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:816/1*1pTLnoOPJKKcKIcRi3q0WA.jpeg" alt="Image Alt" width="500">
</div>

```bash
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))
```


