# -*- coding: utf-8 -*-
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

def main():
    # set data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # set variables
    learning_rate = 1e-3
    training_epocs = 15
    batch_size = 100

    # set place holders
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    # set nodes
    W1 = tf.get_variable("W1", shape=[784, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.get_variable("W2", shape=[256, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[256, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L2, W3) + b3

    # set train node
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epocs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        print('Learning Finished')

        # check accuracy
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
            X: mnist.test.images, Y: mnist.test.labels}))

        # Get one and predict
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction: ", sess.run(
            tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))


if __name__ == "__main__":
    main()
