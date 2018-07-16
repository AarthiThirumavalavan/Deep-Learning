import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
import pandas as pd
from keras.preprocessing import text as keras_text, sequence as keras_seq
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout

#Reading and preparing training data
raw = pd.read_fwf(xtrain_obfuscated.txt,header=None)
xtrain_obfuscated = pd.read_fwf(xtrain_obfuscated.txt, header=None)
ytrain = pd.read_fwf(ytrain.txt, header=None)
xtrain_obfuscated['label']=ytrain[0]
xtrain_obfuscated.rename(columns={0:'text'}, inplace=True)

#Reading test file
xtest_obfuscated = pd.read_fwf(xtest_obfuscated.txt,
                               header=None)
xtest_obfuscated.rename(columns={0:'text'}, inplace=True)

#One-hot encoding on training data
xtrain_encoded = pd.get_dummies(xtrain_obfuscated, columns=['label'])

#Text matrix to be fed into neural network
#List sentences train
train_sentence_list = xtrain_encoded["text"].fillna("unknown").values
list_classes = ["label_0","label_1","label_2",'label_3',"label_4","label_5",
                "label_6","label_7","label_8","label_9","label_10","label_11"]
y = xtrain_encoded[list_classes].values

#List sentences test
test_sentence_list = xtest_obfuscated["text"].fillna("unknown").values

max_features = 20000
maxlen = raw[0].map(len).max()
batch_size=256
num_epoch = 20

#Sequence Generation
tokenizer = keras_text.Tokenizer(char_level = True)
tokenizer.fit_on_texts(list(train_sentence_list))

#Train data
train_list_tokenized = tokenizer.texts_to_sequences(train_sentence_list)
X = keras_seq.pad_sequences(train_list_tokenized, maxlen=maxlen)

#Test data
test_list_tokenized = tokenizer.texts_to_sequences(test_sentence_list)
X_test = keras_seq.pad_sequences(test_list_tokenized, maxlen=maxlen)

#Train test split for training and validation set
any_category_pos = np.sum(y,1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                     test_size = 0.1,
                                                     random_state = 7,
                                                     stratify = any_category_pos)
    
#Model
embedding_vector_length = 256
model = Sequential()
model.add(Embedding(max_features, embedding_vector_length, input_length=maxlen))
model.add(Dropout(0.1))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100, dropout=0.25, recurrent_dropout=0.2))
model.add(Dense(12, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs= num_epoch, 
          batch_size= batch_size, 
          validation_data=(X_valid, y_valid),
          shuffle =True)

#Prediction
y_test = model.predict(X_test)
pred = np.argmax(y_test, axis=1)
y_test_output = pd.DataFrame(data=pred)
np.savetxt(ytest.txt, y_test_output.values, fmt='%d')
