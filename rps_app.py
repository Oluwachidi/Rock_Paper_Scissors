#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


# # Load the Trained model

# In[2]:


model = tf.keras.models.load_model('my_model.hdf5')


# # Write a Header and creat a File uploader

# In[7]:


st.write("""
        # Rock_Paper_Scissor Hand Sign Prediction
        """)

st.write("This is a simple image classification web app to predict rock-paper-scissors hand sign")


# In[8]:


file = st.file_uploader("Please upload an image file", type=["jpg","png"])


# # Preprocess the image the user has uploaded then make prediction.

# In[14]:


def import_and_predict(image_data, model):
    size = (75,75)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_resize = (cv2.resize(img, dsize=(75, 75), interpolation=cv2.INTER_CUBIC))/255.
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]

    prediction = model.predict(img_reshape)

    return prediction    
                    


# In[15]:


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) ==0:
        st.write("It is paper!")
    elif np.argmax(prediction) ==1:
        st.write("It is rock!")
    else:
        st.write("It is scissors!")
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    st.write(prediction)

