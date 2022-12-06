# convert database to if-idf
import numpy as np
import pandas as pd

index = [x for x in range(1, 59)]

spam = pd.read_csv('spambase.csv', names=index, header=None)

# elimino le colonne che, da consegna, non ci interessano
del spam[55], spam[56], spam[57]

print(spam.columns.size)

for i in range(1, 54):
    for j in range(0, spam.index.size):
        spam._set_value(j, i,
                        spam._get_value(j, i) * np.log10(
                            spam.index.size / spam[i].value_counts()[spam._get_value(j, i)]))
print('fine')
spam.to_csv('spam_tfidf.csv')
