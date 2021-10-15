#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataset=pd.read_csv('Linear Regression1.csv')


# In[5]:


dataset


# # Simple linear regression

# In[6]:


x=dataset['SALES'].values.reshape(-1,1)
y=dataset['FACEBOOK'].values.reshape(-1,1)


# In[7]:


plt.figure(figsize=(15,7))
plt.scatter(x,y,c='black')
plt.xlabel("Money spent on NEWSPAPER")
plt.ylabel("SALES")
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=42)


# In[9]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[10]:


y_pred = reg.predict(x_test)
plt.figure(figsize=(16,8))
plt.scatter(x,y, c='black')
plt.plot(
    x_test,
    y_pred,
    c='blue',
    linewidth=2
)
plt.xlabel("Money spent on NEWSPAPER")
plt.ylabel("SALES")
plt.show()


# In[11]:


reg.coef_


# In[12]:


reg.intercept_


# In[13]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[14]:


output=reg.predict([[276.12]])
output


# # Multiple linear regression

# In[15]:


x=dataset.drop(['SALES'],axis=1)
y=dataset['SALES'].values.reshape(-1,1)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=42)


# In[17]:


from sklearn.linear_model import LinearRegression
multiple_reg=LinearRegression()
multiple_reg.fit(x_train,y_train)


# In[19]:


y_pred = multiple_reg.predict(x_test)


# In[20]:


multiple_reg.intercept_


# In[21]:


multiple_reg.coef_


# In[22]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:


print("Enter the amount:")
youtube= float(input("YOUTUBE : "))
facebook = float(input("FACEBOOK : "))
newspaper = float(input("NEWSPAPER : "))
output=multiple_reg.predict([[youtube,facebook,newspaper]])
print("Amount you get Rs{:.2f} sales by Rs{} on YOUTUBE, Rs{} on FACEBOOK and Rs{} on NEWSPAPER."      .format(output[0][0] if output else "not found",youtube,facebook,newspaper))


# In[ ]:




