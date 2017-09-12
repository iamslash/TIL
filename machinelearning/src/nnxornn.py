# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def main():
    # set var
    tf.set_random_seed(777)
    learning_rate = 0.1

    # set data
    x_data = [[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]
    y_data = [[0],
              [1],
              [1],
              [0]]
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    # set in layer 
    X = tf.placeholder(tf.float32, [None, 2]) # ? x 2
    Y = tf.placeholder(tf.float32, [None, 1]) # ? x 1

    # set hidden layer
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    # set out layer
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    # set train
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # set accuracy
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    # Launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            sess.run(train, feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0:
                print(step, sess.run(cost, feed_dict={
                    X: x_data, Y: y_data}), sess.run([W1, W2]))

        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

if __name__ == "__main__":
    main()
