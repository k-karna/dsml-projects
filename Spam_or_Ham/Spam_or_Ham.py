#!/usr/bin/env python
# coding: utf-8

# # This project is to classify Spam or Ham from the SMSSpamCollection

# In[1]:


import nltk


# In[2]:


#nltk.download_shell()

#Used nltk.download_shell() to get identifier 'stopwords'


# In[3]:


messages = [line.rstrip() for line in open('SMSSpamCollection')]

#SMSSpamCollection is dataset taken from UCI


# In[4]:


print(len(messages))


# In[5]:


#Checking any random message
messages[50]


# In[6]:


#Checking first 10 messages to see how label are listed

for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[7]:


#Converting SMSSpamCollection into dataset

import pandas as pd
messages = pd.read_csv('SMSSpamCollection',sep ='\t',names=['label','message'])


# In[8]:


messages.head()


# Exploratory Data Analysis

# In[9]:


messages.describe()


# In[10]:


#Grouping by label to get clear stat of Ham and Spam messages
messages.groupby('label').describe()


# In[11]:


#Checking how long the messages are

messages['length'] = messages['message'].apply(len)


# In[12]:


messages.head()


# In[13]:


#Creating Data Visualization as per length of messages

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


messages['length'].plot.hist(bins=100)


# In[15]:


messages['length'].describe()


# In[16]:


#Checking the lengthiest message of 910 

messages[messages['length'] == 910]['message'].iloc[0]


# In[17]:


#Creating a histogram based on column length by label 'Ham' or 'Spam'

messages.hist(column='length',by='label',bins=60,figsize=(12,4))


# Text Pre-Processing

# In[18]:


#We will use Bag-of-Words approach to convert corpus into vector to perform classfication task
#First we will need to remove punctuation, and common words using Panda's String and NLTK 'stopwords'


import string
from nltk.corpus import stopwords


# In[19]:


def text_process(mess):
    """
    1.remove punc
    2. remove stopwords
    3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower not in stopwords.words('english')]


# In[20]:


#Checking top 10 of our dataframe
messages.head(10)


# In[21]:


#Now Tokenising(converting normal text string into list of words) messages and returning those top 10

messages['message'].head(10).apply(text_process)


# Vectorization

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


#Using built fuction text_process as analyzer of messge column

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[24]:


#Printing total number of vocab_ words
print(len(bow_transformer.vocabulary_))


# In[25]:


#taking one text message and get its bow counts as a vector, putting to use our new bow_transformer:
message5 = messages['message'][4]
print(message5)


# In[26]:


#Checking its vector representation
bow5 = bow_transformer.transform([message5])
print(bow5)
print(bow5.shape)


# In[27]:


#using .transform on our Bag-of-Words(bow) transformed object n transforming the entire DataFrame of messages
messages_bow = bow_transformer.transform(messages['message'])


# In[28]:


#checking how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix:
print('Shape of Sparse Matrix:',messages_bow.shape)
#Checking non-zero occurences in SMS corpus
print('Amount of Non-Zero Occurences:',messages_bow.nnz)


# In[29]:


#Checking Sparsity
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))


# Using TF-IDF

# In[30]:


#We know
#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)


# In[31]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[33]:


#Checking Term Frequency and Inverse Document Frequency for message 5

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf5 = tfidf_transformer.transform(bow5)
print(tfidf5)


# In[37]:


#Converting bow corpus into TFIDF corpus at once

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf)


# Training Model Using Naive-Bayes Algorithm

# In[39]:


from sklearn.naive_bayes import MultinomialNB

#Creating spam_detect_model
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[42]:


#Classifying a message using model, and comparing with expected result
print('Predicted',spam_detect_model.predict(tfidf5)[0])
print('Expected:', messages.label[3])


# Model Evaluation

# In[43]:


#Getting all_predictions for entire dataset using our model

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[44]:


#As we shouldn't be using the same data to evaluate where we trained our model
#So, we will train-test-split
#Then Pipeline of SCikit-Learn for entire workflow so far


# In[54]:


from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.25)


# In[55]:


from sklearn.pipeline import Pipeline


# In[56]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[57]:


#passing message text data to the pipeline to do our text pre-processing 
pipeline.fit(msg_train,label_train)


# In[58]:


predictions = pipeline.predict(msg_test)


# Classification Report and Confusion Matrix

# In[59]:


from sklearn.metrics import classification_report, confusion_matrix


# In[60]:


print(classification_report(label_test,predictions))
print(confusion_matrix(label_test,predictions))


# In[ ]:




