#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


from sklearn.datasets import load_boston
dataset = load_boston()


# In[18]:


X = pd.DataFrame(dataset.data,columns=dataset.feature_names)
y = pd.DataFrame(dataset.target,columns=['output'])


# In[19]:


print(dataset.DESCR)


# In[21]:


X.head()


# In[49]:


X.describe().T


# In[23]:


X.info()


# In[78]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)


# In[79]:


from sklearn.preprocessing import MinMaxScaler
scale_fn = MinMaxScaler(feature_range=(0,1))


# In[77]:


get_ipython().run_line_magic('pinfo', 'MinMaxScaler')


# In[82]:


print(X_train.head())
scale_fn.fit(X_train)
tr_X_train = scale_fn.transform(X_train)
print(tr_X_train)


# In[86]:


tr_X_test = scale_fn.transform(X_test)


# In[87]:


print(tr_X_test)


# In[73]:


from sklearn.linear_model import SGDRegressor


# In[88]:


model_l1 = SGDRegressor(penalty='l1')
model_l2 = SGDRegressor(penalty='l2')


# In[89]:


model_l1.fit(tr_X_train,y_train)
model_l2.fit(tr_X_train,y_train)


# In[76]:


X_train.head()


# In[99]:


y_pred_l1 = model_l1.predict(tr_X_test)
y_pred_l2 = model_l2.predict(tr_X_test)


# In[100]:


y_test['y_pred_l1'] = y_pred_l1
y_test['y_pred_l2'] = y_pred_l2


# In[101]:


y_test


# In[93]:


from sklearn.metrics import mean_absolute_error


# In[97]:


mean_absolute_error(y_test.iloc[:,0], y_test.iloc[:,1])


# In[102]:


mean_absolute_error(y_test.iloc[:,0], y_test.iloc[:,2])


# In[116]:


y_train.shape


# In[114]:


beta.shape
X_trainCFS.shape

