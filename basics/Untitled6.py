#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[68]:


df1 = pd.read_csv(r'Crop_recommendation[1].csv')


# In[131]:


df = df1[(df1['label'] == 'rice') | (df1['label'] == 'maize')]


# In[95]:


df


# In[201]:


plt.xlabel('Temperature')
plt.ylabel('Crop')
plt.plot(x_train['temperature'], y_train, color='red')
plt.scatter(df['temperature'], df['label'])
plt.show()


# In[136]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['label'])

df['label'] = encoder.transform(df['label'])


# In[ ]:


from category_encoders import BinaryEncoder
encoder = BinaryEncoder(col = ['label'])
encoder.fit(df['label'])

df['label'] = encoder.transform(df['label'])


# In[137]:


df


# In[138]:


y = df[['label']]
y


# In[162]:


x = df.drop('label', axis = 'columns')
x


# In[218]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 )


# In[219]:


x_test 


# In[203]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)
result = reg.predict(x_test)


# In[190]:


result


# In[191]:


plt.xlabel('Temperature')
plt.ylabel('Crop')
plt.scatter(x_test['temperature'], y_test, color='red')
plt.plot(x_test['temperature'], result)
plt.show()


# In[175]:


plt.plot(result, y_test)
plt.show()

