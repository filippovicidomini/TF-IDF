import pandas as pd
from sklearn.svm import SVR  #

# Path: spambase.csv
data = pd.read_csv('spambase.csv')
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
#svr_rbf.fit(data.iloc[:, :-1], data.iloc[:, -1])
print(data['1'])
#print(svr_rbf.predict(data.iloc[:, :-1]))


#y_rbf = svr_rbf.fit(data.iloc[:, :-1], data.iloc[:, -1]).predict(data.iloc[:, :-1])

#print("RBF: ", y_rbf)
