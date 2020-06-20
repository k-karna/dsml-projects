#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values


# In[3]:


dataset_train.head()


# In[4]:


training_set


# Feature Scaling

# In[5]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0 ,1))


# In[6]:


training_set_scaled = sc.fit_transform(training_set)


# In[7]:


training_set_scaled


# In[8]:


#creating a data structure with 60 timesteps(based on trends of last 60 days) and 1 output

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])


# In[9]:


X_train,y_train = np.array(X_train), np.array(y_train)


# In[10]:


X_train


# In[11]:


y_train


# In[12]:


#reshaping - to add a dimension in np.array

X_train  = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[13]:


X_train


# Buidling the RNN

# In[14]:


#Importing libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[15]:


#initialising the RNN
regressor = Sequential()


# In[16]:


#Adding the first LSTM layer, #
#and some DRopout regularization to avoid overfitting

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#only two last columns in input_shape, because first one, of observations, is already taken into account

regressor.add(Dropout(rate = 0.2)) #Rate of neurons we wanted to ignore, standard is 20%


# In[17]:


#Adding the second LSTM and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


# In[18]:


#Adding the third LSTM and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


# In[19]:


#Adding the fourth LSTM and Dropout regularization
regressor.add(LSTM(units = 50))
#return_sequences, default is False, so not written, as its last layer
regressor.add(Dropout(rate = 0.2))


# In[20]:


#Adding the output layer
regressor.add(Dense(units = 1))


# Compiling the RNN

# In[21]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[22]:


#Fitting the RNN to the training set

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[23]:


#Making prediction & visualize the result

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values


# In[26]:


#Getting predicted price of Google Stock

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

#For each financial day, we need 60 back days data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


# In[29]:


#Getting thr predicted stock price of 2017
X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[30]:


predicted_stock_price


# In[31]:


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[36]:


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
rmse


# In[ ]:




