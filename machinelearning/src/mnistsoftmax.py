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
    W = tf.Variable(tf.random_normal([784, 10]))
    b = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(X, W) + b

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
