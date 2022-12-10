# file in python to translate a databese to a tf-idf format (see the commented part)
# I think I have to separate the moment in which I calculate the frequency of the word in the database
# and the moment in which I calculate the tf-idf
#
# I have to do this because I have to calculate the frequency of the word in the database only one time,
# and then I have to calculate the tf-idf for each document
#
# need to consider the case in which the word is not in the database




from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

index = range(1, 59)
# I have to read the database
data = pd.read_csv('spambase.csv', names=index, header=None)
print(data.head())
# how to trasform a pandas dataframe in a numpy array?
targhet = data[58]
del data[55], data[56], data[57], data[58]
data = data.values


transformer = TfidfTransformer(smooth_idf=False, norm=None, use_idf=True, sublinear_tf=False)

tfidf = transformer.fit_transform(data)
# how to convert a nupmy array in a pandas dataframe?
tfidf = pd.DataFrame(tfidf.toarray())
print(tfidf.head())