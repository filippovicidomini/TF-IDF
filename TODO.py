import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

number: list = [1, 10, 20, 50, 100, 150, 200, 250, 300, 400]

for i in range(len(number)):
    RFC = RandomForestClassifier(n_estimators=number[i])

    start_time = time.time()
    RFC.fit(X_train, y_train)
    print('Training time: %f' % (time.time() - start_time))
    RFC.score(X_test, y_test)
    start_time = time.time()
    y_predict = RFC.predict(X_test)
    print('Prediction time: %f' % (time.time() - start_time))
    print('Missclassified examples: %d' % (y_test != y_predict).sum())
    print('Accuracy: %.3f' % accuracy_score(y_test, y_predict))