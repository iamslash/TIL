# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class MnistCnn:

    def __init__(self, sess, name, f_learn_rate=1e-3):
        self.sess = sess
        self.name = name
        self.f_learn_rate = f_learn_rate
        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            # set place holder
            self.t_train = tf.placeholder(tf.bool)
            self.t_X = tf.placeholder(tf.float32, [None, 784])
            t_X_img  = tf.reshape(self.t_X, [-1, 28, 28, 1])
            self.t_Y = tf.placeholder(tf.float32, [None, 10])

            # layer 1
            conv1    = tf.layers.conv2d(inputs=t_X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool1    = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.t_train)

            # layer 2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.t_train)

            # layer 3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.t_train)

            # layer4
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.t_train)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.t_H = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_H, labels=self.t_Y))        
        self.t_T = tf.train.AdamOptimizer(learning_rate=self.f_learn_rate).minimize(self.t_C)

        t_pred = tf.equal(tf.argmax(self.t_H, 1), tf.argmax(self.t_Y, 1))
        self.t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))

    def predict(self, l_X, b_train=False):
        return self.sess.run(self.t_H, feed_dict={self.t_X: l_X, self.t_train: b_train})

    def get_accuracy(self, l_X, l_Y, b_train=False):
        return self.sess.run(self.t_accu, feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_train: b_train})

    def train(self, l_X, l_Y, b_train=True):
        return self.sess.run([self.t_C, self.t_T], feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_train: b_train})

def main():
    # set variables
    tf.set_random_seed(777)

    # set data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # set hyper parameters
    n_epochs = 15
    n_batch_size = 100

    # launch nodes
    with tf.Session() as sess:
        m1 = MnistCnn(sess, "m1")
        sess.run(tf.global_variables_initializer())
        print('Learning started!!!')
        for n_epoch in range(n_epochs):
            f_avg_cost = 0
            n_total_batch = int(mnist.train.num_examples / n_batch_size)
            for i in range(n_total_batch):
                l_X, l_Y = mnist.train.next_batch(n_batch_size)
                f_cost, _ = m1.train(l_X, l_Y)
                f_avg_cost += f_cost / n_total_batch
                # if i % 10 == 0:
                #     print("  ", i, "f_avg_cost: ", f_avg_cost)
            print(f'epoch: {n_epoch:10d}, cost: {f_avg_cost:10.9f}')                    
        print('Learning ended!!!')
        print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
    
if __name__ == "__main__":
    main()
