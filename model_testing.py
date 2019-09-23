"""
Author: Dhvani D Patel
Date created: 09/18/2019
Date last modified: 09/23/2019

Age model - https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/
"""

import scipy.io
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from keras.preprocessing import image
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json


# Loading gender model - VGG16
vgg16_model_g = keras.applications.vgg16.VGG16()
model_g = Sequential()
for layer in vgg16_model_g.layers[:-1]:
    model_g.add(layer)
    
for layer in model_g.layers:
    layer.trainable = False
    
model_g.add(Dense(2, activation='softmax'))
model_g.load_weights('gender_wiki_new_weights.h5')


# Loading Age model - VGGFace
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('vgg_face_weights.h5')

#freeze all layers of VGG-Face except last 7 one
for layer in model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

age_model = Model(inputs=model.input, outputs=base_model_output)

age_model.load_weights("age_model_weights.h5")



# This function loads the image from path and performs pre-processing
def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img /= 255
    return test_img


output_indexes = np.array([i for i in range(0, 101)])
# This function predicts age and gender for a given image
def test_imgs(picture):
    
    prediction_a = age_model.predict(loadImage(picture))
    prediction_g = model_g.predict(loadImage(picture))

    img = image.load_img(picture)#, target_size=(224, 224))
    plt.imshow(img)
    plt.show()

    print("Most dominant age class: ",np.argmax(prediction_a))

    # Apparent age is the prediction approach and convert classification task to regression. 
    # Multiply each softmax output with its label and sum of all these numbers is the apparent age.
    apparent_age = np.round(np.sum(prediction_a * output_indexes, axis = 1))
    print("apparent age: ", int(apparent_age[0]))
    gender = "Male" if np.argmax(prediction_g) == 1 else "Female"

    print("gender: ", gender)


# Test for all images in the folder
for i in range(16):
    s = './test/' + str(i+1) + '.jpg'
    test_imgs(s)


# For face detection and age-gender prediction in real-time.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, img = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(img, 1.3, 5)


    for (x,y,w,h) in faces: 
        if w > 130: # Ignoring all small faces
            cv2.rectangle(img,(x,y), (x+w, y+h), (128,128,1))

            # Extract detected face (crop faces)
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] 

            try:
                # Age and gender data set has 40% margin around the face. Expand the detected face.
                margin = 30
                margin_x = int((w*margin)/100); margin_y = int((h*margin)/100)
                detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]

            except:
                print("Detected face has no margin")

            try:
                # VGG model input is (224,244) so resize
                detected_face = cv2.resize(detected_face, (224, 224))

                # Pre-processing images
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                # Predict the apparent age and gender
                age_distributiona = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributiona * output_indexes, axis=1))[0]))
                
                gender_distribution = model_g.predict(img_pixels)[0]
                gender_index = np.argmax(gender_distribution)

                if gender_index == 0: 
                    gender = "Female"
                else:
                    gender = "Male"

                # Background for age gender text box
                info_box_color = (46,200,255)
                triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
                cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                cv2.rectangle(img,(x+int(w/2)-50,y),(x+int(w/2)+75,y-90),info_box_color,cv2.FILLED)
                
                # Put age and gender text in the box
                cv2.putText(img, apparent_age, (x+int(w/2)-5, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
                cv2.putText(img, gender, (x+int(w/2)-42, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

            except Exception as e:
                pass
    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()




