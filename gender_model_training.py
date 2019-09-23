"""
Author: Dhvani D Patel
Date created: 09/16/2019
Date last modified: 09/18/2019
"""

import scipy.io
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from keras.preprocessing import image
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import dlib
from contextlib import contextmanager
import matplotlib.pyplot as plt
from keras.models import load_model


# Loading the imdb-wiki dataset from matlab file with SciPy
mat = scipy.io.loadmat('wiki_crop/wiki.mat')


# Converting to pandas dataframe makes transformation easier
instances = mat['wiki'][0][0][0].shape[1]

columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

df = pd.DataFrame(index = range(0,instances), columns = columns)
for i in mat:
    if i == "wiki":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            df[columns[j]] = pd.DataFrame(current_array[j][0])


# Dataset contains date of birth in matlab datenum format. Convert this to python datatime format. (Only need birthyear)
def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    
    exact_date = datetime.fromordinal(int(datenum)) + timedelta(days=int(days)) + timedelta(hours=int(hours)) + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) - timedelta(days=366)
    
    return exact_date.year
df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)


# Subtracting the year of photo taken with birth year, we get the age
df['age'] = df['photo_taken'] - df['date_of_birth']




'''Data Cleaning'''

# Remove pictures does not include face
df = df[df['face_score'] != -np.inf]
 
# Some pictures include more than one face, remove them
df = df[df['second_face_score'].isna()]
 
# Check threshold
df = df[df['face_score'] >= 3]
 
# Some records do not have a gender information
df = df[~df['gender'].isna()]
df = df.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])

# Some guys seem to be greater than 100. Some of these are paintings. Remove these old guys
df = df[df['age'] <= 100]
 
# Some guys seem to be unborn in the data set
df = df[df['age'] > 0]


# Storing the pixel values of all images
target_size = (224, 224)
def getImagePixels(image_path):
    img = image.load_img("wiki_crop/%s" % image_path[0], grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    #x = preprocess_input(x)
    return x
df['pixels'] = df['full_path'].apply(getImagePixels)


# Applying binary encoding to target gender classes. 0 - Female, 1 - Male.
target = df['gender'].values
target_classes = keras.utils.to_categorical(target, 2)


# Creating a numpy array of image pixels
features = []

for i in range(0, df.shape[0]):
    features.append(df['pixels'].values[i])

features = np.array(features)
features = features.reshape(features.shape[0], 224, 224, 3)
features /= 255 #normalize in [0, 1]


# Spliting the training and validation dataset
x_train, x_valid, y_train, y_valid = train_test_split(features, target_classes, test_size=0.10)


'''
Building the VGG16 model
'''
vgg16_model = keras.applications.vgg16.VGG16()

# The last dense layer has 1000 output nodes, but we only need 2, thus add all other layers except last one.
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
for layer in model.layers:
    layer.trainable = False
    
model.add(Dense(2, activation='softmax'))


# Compiling the model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# Fitting the model (Training the model)
model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), verbose=1)


# Saving the model weights
model.save_weights('gender_wiki_new_weights.h5')




