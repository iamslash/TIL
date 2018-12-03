# -*- coding: utf-8 -*-
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

def main():
    # set data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # set variables
    f_learning_rate = 1e-3
    n_epocs = 15
    n_batch_size = 100

    # set place holders
    t_X = tf.placeholder(tf.float32, [None, 784])
    t_Y = tf.placeholder(tf.float32, [None, 10])

    # set nodes
    t_W1 = tf.Variable(tf.random_normal([784, 256]))
    t_b1 = tf.Variable(tf.random_normal([256]))
    t_L1 = tf.nn.relu(tf.matmul(t_X, t_W1) + t_b1)

    t_W2 = tf.Variable(tf.random_normal([256, 256]))
    t_b2 = tf.Variable(tf.random_normal([256]))
    t_L2 = tf.nn.relu(tf.matmul(t_L1, t_W2) + t_b2)

    t_W3 = tf.Variable(tf.random_normal([256, 10]))
    t_b3 = tf.Variable(tf.random_normal([10]))
    t_H = tf.matmul(t_L2, t_W3) + t_b3

    # set train node
    t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_H, labels=t_Y))
    t_T = tf.train.AdamOptimizer(learning_rate=f_learning_rate).minimize(t_C)

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_epoch in range(n_epocs):
            f_avg_cost = 0
            n_total_batch = int(mnist.train.num_examples / n_batch_size)
            for i in range(n_total_batch):
                l_X, l_Y = mnist.train.next_batch(n_batch_size)
                f_cost, _ = sess.run([t_C, t_T], feed_dict={t_X: l_X, t_Y: l_Y})
                f_avg_cost += f_cost / n_total_batch

            print('Epoch:', '%04d' % (n_epoch + 1), 'cost =', '{:.9f}'.format(f_avg_cost))
        print('Learning Finished')

        # check accuracy
        t_pred = tf.equal(tf.argmax(t_H, 1), tf.argmax(t_Y, 1))
        t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))
        print('Accuracy:', sess.run(t_accu, feed_dict={
            t_X: mnist.test.images, t_Y: mnist.test.labels}))

        # Get one and predict
        n_r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[n_r:n_r + 1], 1)))
        print("Prediction: ", sess.run(tf.argmax(t_H, 1), feed_dict={t_X: mnist.test.images[n_r:n_r + 1]}))


if __name__ == "__main__":
    main()
