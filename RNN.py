# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:51:08 2018

@author: hp
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time


start_time = time.time()
#To evaluate time taken for the training
def elapsed(sec):
    if sec<60:
        return str(sec) + "sec"
    elif sec<(60*60):
        return str(sec/60) + "min"
    
#To store graph
logs_path = 'E:/Python Exercises/Deeplearning_tutorial'
writer = tf.summary.FileWriter(logs_path)#To display graph using tensorboard
#Text file containing words for training
training_file = 'E:/Python Exercises/Deeplearning_tutorial/3wishes.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()#read line by line
    content = [x.strip() for x in content]#remove first and last whitespace
    content = [content[i].split() for i in range(len(content))]#remove all whitespaces present
    content = np.array(content)#creating an array 
    content = np.reshape(content, [-1, ])#reshaping array #-1 to make the reshaped matrix as compatible
    return content
    
training_data = read_data(training_file)
print("Loaded training data...")
#To create a dictionary based on frequency of occurences of each symbol
def build_dataset(words):
    count = collections.Counter(words).most_common()#most common words with their frequency of occurence will be created as a dictionary
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)#to create key value pair of words and their frequency
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))#reverse of dictionary
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

#Parameters
learning_rate = 0.001
training_iters = 50000#epochs
display_step = 1000#batch size
n_input = 3

#number of units in RNN cell
n_hidden = 512

#tf Graph unit
x = tf.placeholder("float", [None, n_input, 1])#input values
y = tf.placeholder("float", [None, vocab_size])#labels

#RNN output node weights and biases
weights = {'out' : tf.Variable(tf.random_normal([n_hidden, vocab_size]))}
biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])#reshape to [1, n_input]
    #generate a n_input sequence of inputs , eg.: [had] [a] [general] -> [20] [6] [33]
    x = tf.split(x, n_input, 1)
    #2layer LSTM, each layer has n_hidden units
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])    
    #generate predictions
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype = tf.float32)
    #there are n_inputs outputs but we want only last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
#Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, label=y))#to get probability of each symbol
optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)

#Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initializing the variables
init = tf.global_variables_initializer()

#Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input+1
    acc_total = 0
    loss_total = 0
    
    writer.add_graph(session.graph)
    
    while step < training_iters:
        #Generate a minibatch and add some randomness to selection process
        if offset < (len(training_data) - end_offset):
            offset = random.randint(0, n_input+1)
            
        symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        
        symbols_out_onehot = np.zeros([vocab_size], dtypes=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                    feed_dict={x: symbols_in_keys, y:symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter=" +str(step+1) + ", Average Loss=" +\
                  "{:.6f}".format(loss_total/display_step) + ", Average accuracy="+\
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset+n_input)]
            symbols_out = training_data[offset+n_input]
            symbols_out_pred = reverse_dictionary[int(tf.arg_max(onehot_pred, 1).eval())]
            print("%s -[%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
        step += 1
        offset += (n_input +1)
        print("Optimization finished!")
        print("Elapsed time:", elapsed(time.time() - start_time))#time taken to train the model
        #to treat exceptions to treat words not in dictionary
        while True:
            prompt = "%s words:" %n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict = {x:keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" %(sentence, reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")
        
        
        
        
        

