#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# In[2]:
import os
import glob

# In[6]:

emails, labels =[], []

# In[7]:
# 데이터셋 
# http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz

paths = ["enron1/spam/", "enron1/ham/"]
for path in paths:
    for filename in glob.glob(os.path.join(path, "*.txt")):
        with open(filename, "r", encoding="ISO-8859-1") as file:
            emails.append(file.read())
            if path.endswith("spam/"):
                labels.append(1)
            else:
                labels.append(0)
print(np.unique(labels, return_counts=True))


# In[8]:


import nltk
nltk.download("names")
nltk.download("wordnet")


# In[9]:


from nltk.corpus import names
from nltk.stem import WordNetLemmatizer


# In[10]:


all_names = set(names.words())


# In[11]:


lemmatizer = WordNetLemmatizer()


# In[12]:


cleaned_emails = []


# In[13]:


for email in emails:
    cleaned_emails.append(" ".join([lemmatizer.lemmatize(word.lower())                                     for word in email.split()                                     if word.isalpha() and word not in all_names]))

//여기까지 오류 안남
# In[14]:
//여기부터 배열 스택오버플로우 오류

cleaned_emails[0]

# In[15]:

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english", max_features=500)
term_docs = vectorizer.fit_transform(cleaned_emails)


# In[16]:


print(term_docs[0])


# In[18]:


term_docs.toarray()


# In[20]:


feature_names = vectorizer.get_feature_names()
len(feature_names)


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels, test_size=0.3, random_state=35)


# In[24]:


len(X_train), len(y_train), len(X_test), len(y_test)


# In[25]:


term_docs_train = vectorizer.fit_transform(X_train)
term_docs_test = vectorizer.transform(X_test)


# In[26]:
//여기부터 12-3 강의 시작, 모델 성능 측정
// 나이브베이즈 모델 적합화 , term docs train 변환된 훈련용 데이터를 사용


from sklearn.naive_bayes import MultinomialNB


# In[27]:


naive_bayes = MultinomialNB(alpha=1, fit_prior=True)
//1이면 스팸 0이면 햄

# In[28]:


naive_bayes.fit(term_docs_train, y_train)


# In[29]:


y_pred = naive_bayes.predict(term_docs_test)


# In[30]:


np.unique(y_pred, return_counts=True)


# In[31]:


naive_bayes.score(term_docs_test, y_test)
//여기서 나이브베이즈 성능 측정 가능

# In[32]:


y_pred_proba = naive_bayes.predict_proba(term_docs_test)
//class 확률 (term docs) -> 스팸의 확률
결과 두개 출력 (햄 확률 / 스팸 확률 )

# In[33]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[38]:

//스팸의 확률
fpr, tpr, _ =roc_curve(y_test, y_pred_proba[:, 1])
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr, "r-", label="MultinomialNB")
plt.plot([0, 1], [0, 1], "b--", label="random guessing")
plt.title("AUC={:.4f}".format(auc))
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.legend()

//나이브베이즈는 범주형 부분에서 강력한 성능발휘
# In[39]:


from sklearn.model_selection import GridSearchCV


# In[47]:


parameters = {
    "alpha": [0.5, 1.0, 1.5, 2.0],
    "fit_prior": [True, False]
}


# In[48]:


grid_search = GridSearchCV(naive_bayes, parameters, n_jobs=-1, cv=10, scoring="roc_auc")


# In[49]:


grid_search.fit(term_docs_train, y_train)


# In[50]:


grid_search.best_params_


# In[51]:


naive_bayes_best = grid_search.best_estimator_


# In[52]:


y_pred_proba = naive_bayes_best.predict_proba(term_docs_test)
y_pred_proba


# In[53]:


fpr, tpr, _ =roc_curve(y_test, y_pred_proba[:, 1])
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr, "r-", label="MultinomialNB")
plt.plot([0, 1], [0, 1], "b--", label="random guessing")
plt.title("AUC={:.4f}".format(auc))
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.legend()


# In[ ]:




