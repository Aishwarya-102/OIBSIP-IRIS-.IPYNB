#!/usr/bin/env python
# coding: utf-8

# In[63]:


df='AISHWARYA PATIL'
print(df)


# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import math
import statistics as sts
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[4]:


df=pd.read_csv(r'C:\Users\aishw\Desktop\Iris.csv')
print(df)


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.size


# In[8]:


df.head()


# In[10]:


df.select_dtypes


# In[11]:


df.select_dtypes('O')


# In[12]:


df.Id


# In[13]:


df.isnull().sum()


# In[15]:


df.describe()


# In[29]:


df


# In[30]:


df['Species'].value_counts()


# In[35]:


sns.pairplot(df)


# In[36]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[37]:


x


# In[38]:


y


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[40]:


x_train.shape


# In[41]:


x_test.shape


# In[44]:


y_train.shape


# In[45]:


y_test.shape


# 

# In[48]:


data=df.values
x=data[:,0:4]
y=data[:,4]


# In[55]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)


# In[57]:


x_new = np.array([[29,5.2,3.4,1.4]])

prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))

