#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


dataset = mnist.load_data("mnistdata.db")


# In[3]:


train , test = dataset


# In[4]:


X_train , y_train = train
X_test , y_test = test


# In[5]:


X_train.shape


# In[6]:


y_train.shape


# In[7]:


y_train


# In[8]:


from keras.utils import to_categorical

y_train_cat = to_categorical(y_train)

y_train_cat


# In[9]:


# flatning
#X_train_rs = X_train.reshape(-1 ,(28*28,1))
#X_test_rs = X_test.reshape(-1, 28*28)


# In[10]:


X_train_rs = X_train.reshape(60000,28,28,1)
X_test_rs = X_test.reshape(10000,28,28,1)


# In[11]:


from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense


# In[12]:


from keras.models import Sequential


# In[13]:


model = Sequential()


# In[14]:


model.add(Convolution2D( filters=32 ,kernel_size = (3,3), activation = 'relu', input_shape=(28,28,1)))


# In[15]:


model.add(MaxPooling2D(pool_size = (2,2)))


# In[16]:


model.add(Flatten())


# In[17]:


model.add(Dense(units = 64 , activation = 'relu'))


# In[18]:


model.add(Dense(units = 32 , activation = 'relu'))


# In[19]:


model.add(Dense(units = 10 , activation = 'softmax'))


# In[20]:


model.summary()


# In[21]:


from keras.optimizers import adam


# In[22]:


model.compile( optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'],)


# In[23]:


model.fit(X_train_rs, y_train_cat,epochs=13)


# In[34]:


model.save("cnn_mnist_model.h5")


# In[24]:


#model.predict(X_test_rs)


# In[25]:


#X_test_rs[0].shape 


# In[26]:


#import numpy as np


# In[27]:


#test_image = X_test_rs[0]


# In[28]:


#test_image = np.expand_dims(test_image, axis=0)


# In[29]:


#test_image.shape


# In[30]:


#model.predict(test_image)


# In[32]:


#plt.imshow(X_test[0])


# In[33]:


#plt.imshow(result)


# In[ ]:




