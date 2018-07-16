# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:23:32 2018

@author: hp
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#SINGLE LAYER PERCEPTRON
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])#desired output

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y_ = tf.matmul(x,w) +b
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_))#y_ -> predicted output

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y:batch[1]})
    
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y,1))#to check if actual prediction matches true value

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#tf.cast converts boolean to float
print(accuracy.eval(feed_dict = {x:mnist.test.images, y:mnist.test.labels}))