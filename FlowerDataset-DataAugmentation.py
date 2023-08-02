#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as py
import cv2
import os
import PIL


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos',origin=dataset_url, cache_dir = ".",untar=True)


# In[4]:


data_dir


# In[5]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[6]:


list(data_dir.glob("*/*.jpg"))


# In[7]:


len(list(data_dir.glob("*/*.jpg")))


# In[8]:


roses = list(data_dir.glob("roses/*"))


# In[9]:


roses[:5]


# In[10]:


PIL.Image.open(str(roses[1]))


# In[19]:


tulips = list(data_dir.glob("tulips/*.jpg"))


# In[20]:


tulips


# In[21]:


PIL.Image.open(str(tulips[0]))


# In[11]:


flowers_images_dict = {
    "roses":list(data_dir.glob("roses/*.jpg")),
    "daisy":list(data_dir.glob("daisy/*.jpg")),
    "dandelion":list(data_dir.glob("dandelion/*.jpg")),
    "sunflowers":list(data_dir.glob("sunflowers/*.jpg")),
    "tulips":list(data_dir.glob("tulips/*.jpg"))
}


# In[12]:


flowers_images_dict['daisy']


# In[13]:


flowers_labels_dict = {
    'roses':0,
    'daisy':1,
    'dandelion':2,
    'sunflowers':3,
    'tulips':4
} 


# In[14]:


img = cv2.imread(str(flowers_images_dict['roses'][0])) 


# In[15]:


img


# In[16]:


img.shape


# In[17]:


cv2.resize(img,(180,180)).shape


# In[18]:


X,y= [],[]

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[19]:


import numpy as np


# In[20]:


X = np.array(X)
y = np.array(y)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)


# In[22]:


len(X_train)


# In[23]:


X_train / 255


# In[24]:


X_train_scaled = X_train/255
X_test_scaled = X_test/255


# In[25]:


model = Sequential([
    layers.Conv2D(16,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dense(5,activation = "softmax")
])
model.compile(optimizer = 'adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


# In[26]:


model.fit(X_train_scaled,y_train,epochs=10)


# In[27]:


model.evaluate(X_test_scaled,y_test)


# In[28]:


predictions = model.predict(X_test_scaled)
predictions


# In[29]:


np.argmax(np.array([0,78,123,8]))


# In[30]:


np.argmax(predictions[0])


# In[31]:


y_test[0]


# In[41]:


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.9), 
])


# In[33]:


py.axis("off")
py.imshow(X[0])


# In[42]:


py.axis("off")
py.imshow(data_augmentation(X)[0].numpy().astype('uint8'))


# In[49]:


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.1),

])


# In[47]:


py.axis("off")
py.imshow(data_augmentation(X)[0].numpy().astype('uint8'))


# In[50]:


model = Sequential([
    data_augmentation,
    layers.Conv2D(16,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dense(5,activation = "softmax")
])
model.compile(optimizer = 'adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


# In[51]:


model.fit(X_train_scaled,y_train,epochs=10)


# In[52]:


model.evaluate(X_test_scaled,y_test)

