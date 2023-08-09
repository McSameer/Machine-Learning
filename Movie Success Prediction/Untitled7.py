import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
# %matplotlib inline

df = pd.read_csv('boxoffice.csv', encoding = 'latin-1')

df.fillna('0', inplace = True)
# plt.hist(df['MPAA'], color = 'red')
# plt.show()
# plt.hist((df['opening_revenue']), color = 'green')
# plt.show()
# plt.hist(df['domestic_revenue'], color = 'blue')
# plt.show()
# plt.hist(df['budget'], color = 'yellow')
# plt.show()
# plt.hist(df['world_revenue'], color = 'crimson')
# plt.show()
# plt.hist(df['domestic_revenue'], color = 'blue')
# plt.show()
# plt.hist(df['opening_theaters'], color = 'red')
# plt.show()
# plt.hist(df['release_days'], color = 'green')
# plt.show()
# plt.hist(df['distributor'], color = 'violet')
# plt.show()
# plt.hist(df['release_days'], color = 'blue')
# plt.show()

df['domestic_revenue'] = df['domestic_revenue'].replace('[\$\,]', '', regex = True).astype(float)
df['opening_revenue'] = df['opening_revenue'].replace('[\$\,]', '', regex = True).astype(float)
df['budget'] = df['budget'].replace('[\$\,]', '', regex = True).astype(float)
df['world_revenue'] = df['world_revenue'].replace('[\$\,]', '', regex = True).astype(float)
df['opening_theaters'] = df['opening_theaters'].replace('[\,]', '', regex = True).astype(float)
df = df.drop(['release_days'], axis = 'columns')

for feature in df:
    
    if type(df[feature][0]) != str:
        print(feature)
        df[feature] = df[feature].replace(to_replace = 0, value = df[feature].mean(axis = 0))
        if feature == 'world_revenue':
            continue
        if (feature == 'budget' or feature == 'opening_revenue'):
            df[feature] = np.log(df[feature])
        df[feature] = scale(df[feature])
df['MPAA'] = df['MPAA'].replace(to_replace = '0', value = 'R')
df['genres'] = df['genres'].replace(to_replace = '0', value = 'Action,Drama,Sci-Fi,Thriller')

# plt.hist(df['MPAA'], color = 'red')
# plt.show()
# plt.hist(df['domestic_revenue'], color = 'blue')
# plt.show()
# plt.hist(df['budget'])
# plt.show()
# plt.hist(df['world_revenue'], color = 'crimson')
# plt.show()
# plt.hist(df['domestic_revenue'], color = 'blue')
# plt.show()
# plt.hist(df['opening_theaters'], color = 'red')
# plt.show()
# plt.hist(df['distributor'], color = 'violet')
# plt.show()
# plt.boxplot(df['domestic_revenue'])

mat = df.corr()

for i in range(len(df['genres'])): 
    df['genres'][i] = df['genres'][i].split(',')
df = pd.get_dummies( df.explode(column = ['genres']), columns=['genres']).groupby('title', as_index=False).sum()

x = df.drop(['world_revenue', 'title'], axis = 'columns')
y = df['world_revenue']

matx = x.corr()

from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# from sklearn.ensemble import RandomForestRegressor
# forest = RandomForestRegressor(n_estimators = 20)
# forest.fit(x_train, y_train)


# from sklearn.svm import SVR
# svc = SVR(kernel='rbf')
# svc.fit(x_train, y_train)
# y_pred = svc.predict(x_test)

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train, y_train)
y_pred = linear.predict(x_test)
y_pred

#from sklearn.tree import DecisionTreeRegressor
#tree = DecisionTreeRegressor()
#tree.fit(x_train, y_train)
#y_pred = tree.predict(x_test)
#print(y_pred)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)

print('error = ', mean_absolute_error(y_test, y_pred))
print('sq error = ', metrics.mean_squared_error(y_test, y_pred))
print('accuracy = ', 1 - mean_absolute_error(y_test, y_pred))
print("The accuracy of our model is ", score *100)