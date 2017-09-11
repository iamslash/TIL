# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

def main(lr):
    # set training data set
    x_data = [[1, 2, 1],
              [1, 3, 2],
              [1, 3, 4],
              [1, 5, 5],
              [1, 7, 5],
              [1, 2, 5],
              [1, 6, 6],
              [1, 7, 7]]
    y_data = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0]]
    # set test data set
    x_test = [[2, 1, 1],
              [3, 1, 2],
              [3, 3, 4]]
    y_test = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1]]

    # set nodes
    X = tf.placeholder("float", [None, 3])
    Y = tf.placeholder("float", [None, 3])
    W = tf.Variable(tf.random_normal([3, 3]))
    b = tf.Variable(tf.random_normal([3]))
    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=lr).minimize(cost)

    # set accuracy
    prediction = tf.arg_max(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # Launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(201):
            cost_val, W_val, _ = sess.run(
                [cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
            print(step, cost_val, W_val)

        # predict
        print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
        print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

if __name__ == "__main__":
    main(1.5)
    main(1e-10)
    main(1e-5)
