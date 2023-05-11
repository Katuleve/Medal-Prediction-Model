#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


teams = pd.read_csv('desktop/teams.csv')
teams.head()


# # Selecting your data

# In[13]:


teams = teams[['team', 'country', 'year', 'athletes', 'age', 'prev_medals', 'medals']]

teams.head()


# # Correlation of Medals with other data features

# In[14]:


teams.corr()['medals']


# In[20]:


import seaborn as sns

sns.lmplot(x = 'athletes', y = 'medals', data = teams, fit_reg=True, ci = None)


# # Treating Null Values

# In[23]:


teams[teams.isnull().any(axis = 1)]


# In[24]:


teams = teams.dropna()


# In[25]:


teams


# # Selecting Train and Test Data

# In[26]:


train = teams[teams['year'] < 2012].copy()
test = teams[teams['year'] >= 2012].copy()


# In[30]:


train.shape


# In[31]:


test.shape


# In[32]:


predictors = ['athletes', 'prev_medals']
target = ['medals']


# In[34]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[35]:


lr.fit(train[predictors], train[target])


# In[37]:


predictions = lr.predict(test[predictors])
predictions


# In[38]:


test['predictions'] = predictions


# In[39]:


test


# In[42]:


test.loc[test['predictions'] <0, 'predictions'] = 0
test['predictions'] = test['predictions'].round()
test


# # Defining the Model Error

# In[43]:


from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test['medals'], test['predictions'])

error


# In[44]:


teams.describe()['medals']


# # Comparing our predictions to actual Data

# In[47]:


test[test['team'] == 'USA']


# In[48]:


test[test['team'] == 'KEN']


# # Absolute Error between our prediction and Real Data

# In[52]:


errors = (test['medals'] - test['predictions']).abs()


# In[53]:


errors


# In[55]:


error_by_team = errors.groupby(test['team']).mean()


# In[56]:


error_by_team


# In[59]:


medals_by_team = test['medals'].groupby(test['team']).mean()


# In[60]:


medals_by_team


# In[61]:


error_ratio = error_by_team / medals_by_team


# In[62]:


error_ratio


# In[63]:


error_ratio[~pd.isnull(error_ratio)]


# In[65]:


error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio


# In[ ]:




