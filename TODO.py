import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

spam_data = pd.read_csv('spam_tfidf.csv', index_col=[0])
# spam_data.head()
# data processing
X = spam_data.drop('targhet', axis=1)
y = spam_data['targhet']  # colonna che segna se è o meno spam

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# qui faccio il learning
# voglio fare cross validation usando vari valori di C
c: list = [1.0, 10.0, 100.0, 1000.0, 10000.0, 10000.0]
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

for i in range(0, 6):
    clf = SVC(C=c[i], kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=['ham', 'spam'], columns=['ham', 'spam'])
    sn.heatmap(df_cm, annot=True, ax=axs[int(i / 3), i % 3])
    axs[int(i / 3), i % 3].set_title('C = ' + str(c[i]))
    plt.pause(0.05)