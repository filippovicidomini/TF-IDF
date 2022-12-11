# file in python to translate a databese to a tf-idf format (see the commented part)
# I think I have to separate the moment in which I calculate the frequency of the word in the database
# and the moment in which I calculate the tf-idf
#
# I have to do this because I have to calculate the frequency of the word in the database only one time,
# and then I have to calculate the tf-idf for each document
#
# need to consider the case in which the word is not in the database

import math

import pandas as pd

# create and array of the indexes in order to use it as the firts row of the df which has to identify each column
indexes = range(1, 59)

# read the file and store the dataframe
df = pd.read_csv('spambase.csv', names=indexes)

# we are not interested in the columns 55-57 so we delete those and store the spam/ham informations in a list
del df[55], df[56], df[57]
spam_ham = df[58].tolist()
print(df.head())
# restore the DF in order to use it for the calculation of weights
df_touse = pd.read_csv('spambase.csv', names=indexes)
del df_touse[55], df_touse[56], df_touse[57]

lista = df_touse.values.tolist()

N = 4601


# this function calculates the WEIGHT for each term 'i' in the document 'j'
def TF_w_calculator(lista, i, j):
    dfi = 0
    for k in range(len(lista)):
        if lista[k][i] != 0:
            dfi += 1
    tf_ij = lista[j][i]

    w_ij = tf_ij * math.log(N / dfi)  # usiamo base e
    return w_ij


# create a dataframe similar to the previous one with the WEIGHTs instead of the frequencies
def dataweight(lista):
    list_to_df = []
    for j in range(len(lista)):
        aux_list = []
        for i in range(len(lista[j])):
            aux_list.append(TF_w_calculator(lista, i, j))
        list_to_df.append(aux_list)
    return list_to_df


weight_list = dataweight(lista)
indexes_2 = range(1, 56)
weight_df = pd.DataFrame(weight_list, columns=indexes_2)
del weight_df[55]
weight_df['targhet'] = spam_ham
print(weight_df.head())
weight_df.to_csv('spam_tfidf.csv', index=indexes_2)
