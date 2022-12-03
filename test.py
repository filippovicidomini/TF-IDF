import pandas as pd
from sklearn.svm import SVR  #

# Path: spambase.csv
data = pd.read_csv('spambase.csv')
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)