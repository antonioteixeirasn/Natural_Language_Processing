#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Importando o dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# o texto é separado por tabulação, por isso o segundo parâmetro.
# o terceiro parâmetro serve para ignorar as aspas que aparecem no dataset

dataset.head()


# In[3]:


# Limpando os textos

import re # Biblioteca com tools para limpar os textos (simplificar)
import nltk # Biblioteca para ignorar palavras que são irrelevantes (the, a, and, an, etc)
nltk.download('stopwords') # Baixa as stopwords (irrelevantes para predição)
from nltk.corpus import stopwords # Importa as stopwords para o notebook

from nltk.stem.porter import PorterStemmer
''' esse ultima linha de código serve para realizar o stemmer, ou seja,
simplifica as palavras. por exemplo, loved e love significam a mesma coisa,
assim, transforma as conjugações temporais na forma infinitiva da palavra.
'''

corpus = [] # Cria uma lista com as diferentes reviews, após processo de cleaning

for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) # Transforma pontuação em "espaço"
  review = review.lower() # transforma letras maiúsculas em minúsculas
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  # A ultima linha de código aplica o stemming à lista de palavras
  review = ' '.join(review)
  corpus.append(review)


# In[4]:


# Criando um modelo de Bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# In[5]:


# Dividindo os dados entre treino e teste

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[6]:


# Treinando um modelo Naive Bayes com os dados de treino

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[7]:


# Prevendo os resultados de teste

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[8]:


# Criando a matriz de confusão

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

