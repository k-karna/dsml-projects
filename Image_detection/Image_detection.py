#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


tf.__version__


# Data Preprocessing - two parts - Preprocessing Training Set and Test Set

# In[6]:


#Preprocessing the trainng set

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# In[7]:


#Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# Building the CNN - Initializing the CNN

# In[8]:


cnn = tf.keras.models.Sequential()


# In[9]:


#Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,activation= 'relu', input_shape=[64,64,3]))


# In[10]:


#Pooling(MaxPooling)
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# In[11]:


#Second layer of Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,activation= 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

#input_shape from convolution has been removed as it automatically connects to the first layer


# In[12]:


#Flattening

cnn.add(tf.keras.layers.Flatten())


# In[13]:


#Full Connecting

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# In[14]:


#creating output layer

cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# Training the CNN - Compiling the CNN

# In[15]:


cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[17]:


cnn.fit(x = training_set, validation_data = test_set,steps_per_epoch = 250, epochs = 25)


# In[20]:


#getting result on a single image to check accuracy

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                           target_size = (64, 64))


# In[21]:


test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)


# In[ ]:




