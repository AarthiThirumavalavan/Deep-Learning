# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:34:15 2018

@author: hp
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'E:/Python Exercises/Deeplearning_tutorial/train'
TEST_DIR = 'E:/Python Exercises/Deeplearning_tutorial/test'
IMG_SIZE = 50
LR = 1e-3#0.001
MODEL_NAME = 'dogs-vs-cats-covnet'

#ENCODE THE LABEL FOR CAT AND DOG
def label_img(img):
    #Create one hot encoded vector from image name eg:dog.93.png
    word_label = img.split('.')[-3]
    if word_label == "cat":
        return np.array([1,0])
    elif word_label == "dog":
        return np.array([0,1])
#RESIZING IMAGE TO 50X50 AND CONVERTING IT TO GRAYSCALE   
#Splitting data into training and testing parts
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

#If dataset is not created
train_data = create_train_data()

#MODEL BUILDING
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')#resized image to 50x50x1 matrix
convnet = conv_2d(convnet, 32, 5, activation = 'relu')#32 filters, stride=5, activation=relu
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet, 0.8)#model actually finishes here
convnet = fully_connected(convnet, 2, activation = 'softmax')
convnet = regression(convnet, optimizer ='adam', learning_rate=LR, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data[:500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)#0th element is the image data
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

#test_data = process_test_data()

model.fit(X, Y, n_epoch=10, validation_set=(test_x, test_y),#To calculate accuracy
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir=foo:E:\Python Exercises\Deeplearning_tutorial\log
fig = plt.figure(figsize=(6,12))

test_data = np.load('test_data.npy')
#Feeding random input from test data
for num, data in enumerate(test_data[:16]):
    img_name = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out)==1:
        str_label = "Dog"
        
    else:
        str_label = "Cat"
    
    y.imshow(orig, cmap="gray")
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show