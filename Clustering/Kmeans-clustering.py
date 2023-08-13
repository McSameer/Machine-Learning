import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

df = pd.read_csv('company_dataset.csv')
df = df.iloc[:, 1:]

df = df.dropna()
df = df.drop_duplicates(subset=['name'])
y = df['employees']
df = df.drop(['employees', 'name'], axis = 'columns')

df = df.reset_index(drop = True)
df['review_count'] = df['review_count'].replace('[/Reviews/\(\)]', '', regex = True)

for i in range(len(df['review_count'])):
    if df['review_count'][i][-2] == 'k':
        df['review_count'][i] = float(df['review_count'][i][:-2]) * 1000
    else:
        df['review_count'][i] = float(df['review_count'][i])
        
df['years'] = df['years'].replace('[/years old/]', '', regex = True).astype(int)
df['hq'] = df['hq'].replace('[/ more/]', '', regex = True)

for i in range(len(df['hq'])):
    val = re.findall('[0-9]+', df['hq'][i])
    if len(val) != 0:
        # print(val)
        df['hq'][i] = int(val[0]) + 1
    else:
        df['hq'][i] = 1

df = pd.get_dummies(df, columns = ['ctype'])

df['ratings'] = scale(df['ratings'])
df['review_count'] = scale(df['review_count'])
df['years'] = scale(df['years'])
df['hq'] = scale(df['hq'])

from sklearn.cluster import KMeans
    
km = KMeans(n_clusters=14)
km.fit(df)
y_pred = km.predict(df)
print(np.bincount(y_pred ))

for i in range(0, 5):
    for j in range(i+1, 5):
        plt.xlabel(df.columns[i])
        plt.ylabel(df.columns[j])
        plt.ylim(0, 10)
        plt.scatter(df.values[:, i], df.values[:, j], c = y_pred)
        plt.show()
