# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def main():
    # set var
    tf.set_random_seed(777)  # for reproducibility
    learning_rate = 0.01
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
    X = tf.placeholder(tf.float32, [None, 2], name='x-input')
    Y = tf.placeholder(tf.float32, [None, 1], name='y-input')
    # set hidden layer1
    with tf.name_scope("layer1") as scope:
        W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
        b1 = tf.Variable(tf.random_normal([2]), name='bias1')
        layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

        w1_hist = tf.summary.histogram("weights1", W1)
        b1_hist = tf.summary.histogram("biases1", b1)
        layer1_hist = tf.summary.histogram("layer1", layer1)
    # set hidden layer 1
    with tf.name_scope("layer2") as scope:
        W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
        b2 = tf.Variable(tf.random_normal([1]), name='bias2')
        hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

        w2_hist = tf.summary.histogram("weights2", W2)
        b2_hist = tf.summary.histogram("biases2", b2)
        hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

    # set cost node
    with tf.name_scope("cost") as scope:
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                               tf.log(1 - hypothesis))
        cost_summ = tf.summary.scalar("cost", cost)
    # set train node
    with tf.name_scope("train") as scope:
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # set accuracy node
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    accuracy_summ = tf.summary.scalar("accuracy", accuracy)

    # Launch nodes
    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
        writer.add_graph(sess.graph)  # Show the graph
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, global_step=step)

            if step % 100 == 0:
                print(step, sess.run(cost, feed_dict={
                    X: x_data, Y: y_data}), sess.run([W1, W2]))

        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

if __name__ == "__main__":
    main()
