#!/usr/bin/env python
# coding: utf-8

# # TASK-13

# # One Hot Encoding using sklearn
# One-Hot encoding is a technique of representing categorical data in the form of binary vectors
# One-Hot Encoring has a two step process.
# 
# **Conversion of Categories to Integers**---->Use **LabelEncoder** to convert categories into integers
# **Conversion of Integers to Binary vectors**----->Use **OneHotEncoder** to convert the integers into One-Hot vectors (binary vectors).

# In[50]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[51]:


startup=pd.read_csv('50_Startups.csv')
startup.head()


# In[52]:


startup['State'].value_counts()


# ## Using LabelEncoder to convert Categories into Integers

# In[53]:


le=LabelEncoder()


# In[54]:


state_numeric=le.fit_transform(startup['State'])
state_numeric.shape


# ## Using OneHotEncoder to convert Integers into Binary

# #### OneHotEncoder only works on data that is in column format. To use the integer encoding from LabelEncoder we will have to reshape the output before providing it as an input to OneHotEncoder.

# In[55]:


state_numeric_reshape=state_numeric.reshape(len(state_numeric),1)


# In[56]:


one_hot=OneHotEncoder(sparse=False)


# In[57]:


state_onehot=one_hot.fit_transform(state_numeric_reshape)
state_onehot


# In[58]:


feature_labels=one_hot.categories_


# In[62]:


state_df=pd.DataFrame(state_onehot,columns=['State_C','State_F','State_N'])


# In[63]:


startup=startup.join(state_df)


# In[65]:


startup.head()


# In[67]:


del startup['State']


# In[68]:


startup.head()


# In[ ]:





# In[ ]:





# In[ ]:




