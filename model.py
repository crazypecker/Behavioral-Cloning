
# coding: utf-8

# In[27]:

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn

lines = []

with open('./mydata2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#print(lines[34].split('/'))
images = []
imagesc = []
imagesl = []
imagesr = []
measurements = []

for line in lines:
    # reading center camera images
    source_path_center = line[0]
    tokensc = source_path_center.split('/')
    filename = tokensc[-1]
    local_path = 'mydata2/IMG/'+filename
    imagec = cv2.imread(local_path)
    #imagec = cv2.resize(imagec,(200,66))
    imagesc.append(imagec)
    # reading left camera images
    source_path_left = line[1]
    tokensl = source_path_left.split('/')
    filename = tokensl[-1]
    local_path = 'mydata2/IMG/'+filename
    imagel = cv2.imread(local_path)
    #imagel = cv2.resize(imagel,(200,66))
    imagesl.append(imagel)
    # reading right camera images
    source_path_right = line[2]
    tokensr = source_path_right.split('/')
    filename = tokensr[-1]
    local_path = 'mydata2/IMG/'+filename
    imager = cv2.imread(local_path)
    #imager = cv2.resize(imager,(200,66))
    imagesr.append(imager)
    #reading steeting angle
    measurement = line[3]
    measurements.append(measurement)

#X_train = np.array(imagesc)
#X_left = np.array(imagesl)
#X_right = np.array(imagesr)
#Y_train = np.array(measurements)

augmented_images = []
augmented_measurements = []

for image, measurement in zip(imagesc, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

# TODO: Normalize the images
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

X_train = normalize_grayscale(X_train)


# In[28]:

from keras.models import Sequential

# TODO: Build a Multi-layer feedforward neural network with Keras here.
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout 
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Implementation of NVIDIA proposed architecture as our guiding model
model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3), output_shape=(66,200,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2),activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2),activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2),activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dense(1))
model.summary()



# In[29]:

model.compile('adam', 'mse', ['accuracy'])
history = model.fit(X_train, Y_train, nb_epoch=3, validation_split=0.2, shuffle =True)

model.save('model.h5')


# In[ ]:



