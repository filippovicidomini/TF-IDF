# implementation of Naive Bayes Gaussian Classifier no sklearn

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('spam_tfidf.csv', index_col=[0])
# dataset.head()
dataset[str(1)]

X = dataset.drop('targhet', axis=1)
y = dataset['targhet']  # colonna che segna se è spam o meno
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


def norm_single(param, mean, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-0.5 * (param - mean) ** 2 / (sigma ** 2))


class NaiveBayesGaussian(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):

        self.mean = []  # primo media di ogni singola parola # uno per spam un per Ham
        self.var = []

        self.freq = []
        self.classes = np.unique(y)
        for c in self.classes:
            self.freq.append((y == c).sum() / y.shape[0])
            self.mean.append(X[y == c].mean(axis=0))
            self.var.append(X[y == c].var(axis=0))
        for i in range(len(self.var)):
            self.var[i] += np.min(self.var) * 0.1

    def predict(self, X):

        size = X.shape[0]

        y = np.zeros(size, dtype=self.classes.dtype)
        probs = np.zeros(len(self.classes))

        for i in range(size):
            max_prob = 0
            max_c = 0
            for c in range(len(self.classes)):
                probs = float(self.norm(X.values[i], c) * self.freq[c])

                if probs > max_prob:
                    max_prob = probs
                    max_c = c
            y[i] = self.classes[max_c]
        return y

    # return the profuct of the probabilities of each word in the sentence
    def norm(self, new_doc: list, target: int):
        new_doc = np.array(new_doc)
        prob = 1
        for i in range(len(new_doc)):
            prob *= norm_single(new_doc[i], self.mean[target][i], self.var[target][i])
        return prob


nbg = NaiveBayesGaussian()
nbg.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

fig = plt.figure()
list_of_predictions = []
# for i in range(1):
nbg = NaiveBayesGaussian()
start_time = time.time()
nbg.fit(X_train, y_train)
print('Training time: %f' % (time.time() - start_time))
start_time = time.time()
y_pred_nbg = nbg.predict(X_test)
list_of_predictions.append(y_pred_nbg)
print('Prediction time: %f' % (time.time() - start_time))
print('Missclassified examples: %d' % (y_test != y_pred_nbg).sum())
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_nbg))
score = cross_val_score(nbg, X, y, cv=10, scoring='accuracy')
print(score)
plt.title('Accuracy of Naive Bayes \n Gaussian Classifier: %f' % score.mean())

cm = confusion_matrix(y_test, y_pred_nbg)
sn.heatmap(cm, annot=True, cmap=sn.color_palette("blend:#7AB,#EDA", as_cmap=True))
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.tight_layout()

plt.savefig('confusion matric gaussianNB.png')
