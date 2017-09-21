# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)

def make_lstm_cell(hidden_size):
    return rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

def main():
    # set characters
    sentence = ("if you want to build a ship, don't drum up people together to "
                "collect wood and don't assign them tasks and work, but rather "
                "teach them to long for the endless immensity of the sea.")
    char_set = list(set(sentence))
    char_dic = {w: i for i, w in enumerate(char_set)}

    # set hyper params
    data_dim = len(char_set)
    hidden_size = len(char_set)
    num_classes = len(char_set)
    sequence_length = 10
    learning_rate = 0.1

    # set data
    dataX = []
    dataY = []
    for i in range(0, len(sentence) - sequence_length):
        x_str = sentence[i: i + sequence_length]
        y_str = sentence[i+1: i + sequence_length + 1]
        # print(i, x_str, '->', y_str)
        x = [char_dic[c] for c in x_str]
        y = [char_dic[c] for c in y_str]
        dataX.append(x)
        dataY.append(y)
    batch_size = len(dataX)

    # set placeholder
    X = tf.placeholder(tf.int32, [None, sequence_length])
    Y = tf.placeholder(tf.int32, [None, sequence_length])
    X_one_hot = tf.one_hot(X, num_classes)
    # print(X_one_hot)

    # set rnn
    multi_cells = rnn.MultiRNNCell([make_lstm_cell(hidden_size) for _ in range(2)], state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

    # set FC layer
    X_for_fc = tf.reshape(outputs, [-1, hidden_size])
    outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
    outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
    weights = tf.ones([batch_size, sequence_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(
        logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            _, l, results = sess.run(
                [train, loss, outputs], feed_dict={X: dataX, Y: dataY})
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                print(i, j, ''.join([char_set[t] for t in index]), l)

        results = sess.run(outputs, feed_dict={X: dataX})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            if j is 0:
                print(''.join([char_set[t] for t in index]), end='')
            else:
                print(char_set[index[-1]], end='')

if __name__ == "__main__":
    main()
