# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 12:37:27 2018

@author: hp
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
import pandas as pd
from keras.preprocessing import text as keras_text, sequence as keras_seq
from sklearn.model_selection import train_test_split

#Preparing training data
raw = pd.read_fwf(r'D:/sap/offline_challenge_to_send/xtrain_obfuscated.txt', header=None)
xtrain_obfuscated = pd.read_fwf(r'D:/sap/offline_challenge_to_send/xtrain_obfuscated.txt', header=None)
ytrain = pd.read_fwf(r'D:/sap/offline_challenge_to_send/ytrain.txt',header=None)
xtrain_obfuscated['label']=ytrain[0]
xtrain_obfuscated.rename(columns={0:'text'}, inplace=True)

#Reading test file
xtest_obfuscated = pd.read_fwf(r'D:/sap/offline_challenge_to_send/xtest_obfuscated.txt',header=None)
xtest_obfuscated.rename(columns={0:'text'}, inplace=True)

#One-hot encoding on training data
xtrain_encoded = pd.get_dummies(xtrain_obfuscated, columns=['label'])

#df_encoded_copy=df_encoded.copy()

#List sentences train
#Text matrix to be fed into neural network
train_sentence_list = xtrain_encoded["text"].fillna("unknown").values
list_classes = ["label_0","label_1","label_2",'label_3',"label_4","label_5","label_6","label_7","label_8","label_9","label_10","label_11"]
y = xtrain_encoded[list_classes].values

#List sentences test
test_sentence_list = xtest_obfuscated["text"].fillna("unknown").values

max_features = 20000
maxlen = raw[0].map(len).max()
batch_size=32

#Sequence Generation
tokenizer = keras_text.Tokenizer(char_level = True)
tokenizer.fit_on_texts(list(train_sentence_list))
# train data
train_list_tokenized = tokenizer.texts_to_sequences(train_sentence_list)
X = keras_seq.pad_sequences(train_list_tokenized, maxlen=maxlen)

X_train, X_valid= train_test_split(X, test_size=0.2)
y_train, y_valid= train_test_split(y, test_size=0.2)
# test data
test_list_tokenized = tokenizer.texts_to_sequences(test_sentence_list)
X_test = keras_seq.pad_sequences(test_list_tokenized, maxlen=maxlen)


#Model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(12, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))