# -*- coding: utf-8 -*-
"""
Spam Filtering using the Spamassasinpublic dataset
Classification using Rsndom Forrest and XGBoost

@author: Osi
"""

import os
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost

# Create directories for Ham & Spam datasets
spamPath = os.path.join("spamassassin-public-corpus","spam")
hamPath = os.path.join("spamassassin-public-corpus","ham")


# Create labels for the 2 classes Ham or Spam
corpus = []
labels = []

file_types_and_labels = [(spamPath, 0), (hamPath, 1)]

for filesPath,label in file_types_and_labels:
    files = os.listdir(filesPath)
    for file in files:
        file_path = filePath = os.path.join(filesPath, file)
        
        try:
            with open(file_path, 'r') as myfile:
                data = myfile.read().replace('\n', '')
            data = str(data)
            corpus.append(data)
            labels.append(label)
        except:
            pass
        
        
# Train test split of each dataset
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.33, random_state=42)


# Train NLP pipeline on the training data
text_clf = Pipeline([('vect', HashingVectorizer(input='content', ngram_range=(1, 3))),('tfidf', TfidfTransformer(use_idf=True, )),('rf',RandomForestClassifier(class_weight='balanced')),])

# Classification with XGBoost
#text_clf = Pipeline([('vect', HashingVectorizer(input='content', ngram_range=(1, 3))),('tfidf', TfidfTransformer(use_idf=True, )),('rf',XGBClassifier(class_weight='balanced')),])

text_clf.fit(X_train, y_train)

# Support Vector Classifier
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)


# Evaluate 
print(text_clf.score(X_train,y_train))


print(f'Missclassified samples: {(y_test != y_pred).sum()}')
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
