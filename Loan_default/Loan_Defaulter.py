#!/usr/bin/env python
# coding: utf-8

# To classify and predict whether or not the borrower paid back their loan in full 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loans = pd.read_csv('loan_data.csv')


# In[3]:


loans.head()


# In[4]:


loans.info()


# Exploratory Data Analysis

# In[5]:


#Creating a histogram of two FICO distributions on top of each other, one for each credit.policy outcome

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy = 1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label = 'Credit Policy = 0',alpha=0.6)
plt.legend()
plt.xlabel('FICO')


# In[6]:


#Creating a similar one, just this time select by the not.fully.paid colum

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='Not Fully Paid = 1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label = 'Not Fully Paid = 0',alpha=0.6)
plt.legend()
plt.xlabel('FICO')


# In[7]:


#Creating a countplot using seaborn showing the counts of loans by purpose, 
#with the color hue defined by not.fully.paid.

plt.figure(figsize=(12,6))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# In[8]:


#Creating jointplot to see the trend between FICO score and interest rate

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[9]:


#Create the lmplots to see if the trend differed between not.fully.paid and credit.policy.
    
    
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
          col='not.fully.paid',palette='Set1')


# Setting the Data

# In[10]:


loans.info()


# In[27]:


#As purpose column is categorical, we need to transform it using dummy variables 
#and then we will create a new final dataframe with separate columns for all categorical features of 'purpose'


categorical_feature = ['purpose']


# In[28]:


final_data = pd.get_dummies(loans,columns=categorical_feature,drop_first=True)


# In[29]:


final_data.info()


# Spliting final data into training and test set

# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# Training a Decision Tree Model

# In[16]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[19]:


predictions = dtree.predict(X_test)


# Evaluating Decision Tree Model

# In[20]:


from sklearn.metrics import classification_report,confusion_matrix


# In[21]:


print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))


# # Training a Random Forest Model

# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


rfc = RandomForestClassifier(n_estimators=300)


# In[24]:


rfc.fit(X_train,y_train)


# In[25]:


predictions = rfc.predict(X_test)


# In[26]:


print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))


# In[30]:


#Random Forest did bit well overall than Decision Tree, however recall and f1-score isn't that great for 1


# In[ ]:




