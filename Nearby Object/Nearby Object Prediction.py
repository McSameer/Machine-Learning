import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import scale

df = pd.read_csv('neo.csv')

x = df.drop(['name', 'id', 'est_diameter_max', 'sentry_object', 'orbiting_body'], axis = 'columns')

for feature in x:
    if feature != 'hazardous':
        x[feature] = scale(x[feature])
mat = x.corr()

y = df['hazardous']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

# from sklearn.linear_model import LogisticRegression
# logistic = LogisticRegression()
# logistic.fit(x_train, y_train)
# y_pred = logistic.predict(x_test)

# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier()
# tree.fit(x_train, y_train)
# y_pred = tree.predict(x_test)

print('Training Accuracy = ', svc.score(x_train, y_train))
print('Testing Accuracy = ', svc.score(x_test, y_test))
print('Accuracy Score = ', metrics.accuracy_score(y_test, y_pred))