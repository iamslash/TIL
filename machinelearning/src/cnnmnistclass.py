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
            self.t_K = tf.placeholder(tf.float32)
            self.t_X = tf.placeholder(tf.float32, [None, 784])
            t_X_img  = tf.reshape(self.t_X, [-1, 28, 28, 1])
            self.t_Y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape=(?, 28, 28, 1)
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            t_W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            t_L1 = tf.nn.conv2d(t_X_img, t_W1, strides=[1, 1, 1, 1], padding='SAME')
            t_L1 = tf.nn.relu(t_L1)
            t_L1 = tf.nn.max_pool(t_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            t_L1 = tf.nn.dropout(t_L1, keep_prob=self.t_K)

            # L2 ImgIn shape=(?, 14, 14, 32)
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            t_W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            t_L2 = tf.nn.conv2d(t_L1, t_W2, strides=[1, 1, 1, 1], padding='SAME')
            t_L2 = tf.nn.relu(t_L2)
            t_L2 = tf.nn.max_pool(t_L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            t_L2 = tf.nn.dropout(t_L2, keep_prob=self.t_K)

            # L3 ImgIn shape=(?, 7, 7, 64)
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128)
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            t_W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            t_L3 = tf.nn.conv2d(t_L2, t_W3, strides=[1, 1, 1, 1], padding='SAME')
            t_L3 = tf.nn.relu(t_L3)
            t_L3 = tf.nn.max_pool(t_L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            t_L3 = tf.nn.dropout(t_L3, keep_prob=self.t_K)
            t_L3_flat = tf.reshape(t_L3, [-1, 128 * 4 * 4])

            # L4 FC 4x4x128 inputs -> 625 outputs
            t_W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
            t_b4 = tf.Variable(tf.random_normal([625]))
            t_L4 = tf.nn.relu(tf.matmul(t_L3_flat, t_W4) + t_b4)
            t_L4 = tf.nn.dropout(t_L4, keep_prob=self.t_K)

            # L5 Final FC 625 inputs -> 10 outputs
            t_W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            t_b5 = tf.Variable(tf.random_normal([10]))
            self.t_H = tf.matmul(t_L4, t_W5) + t_b5

        # define cost/loss & optimizer
        self.t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_H, labels=self.t_Y))        
        self.t_T = tf.train.AdamOptimizer(learning_rate=self.f_learn_rate).minimize(self.t_C)

        t_pred = tf.equal(tf.argmax(self.t_H, 1), tf.argmax(self.t_Y, 1))
        self.t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))

    def predict(self, l_X, f_K=1.0):
        return self.sess.run(self.t_H, feed_dict={self.t_X: l_X, self.t_K: f_K})

    def get_accuracy(self, l_X, l_Y, f_K=1.0):
        return self.sess.run(self.t_accu, feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_K: f_K})

    def train(self, l_X, l_Y, f_K=0.7):
        return self.sess.run([self.t_C, self.t_T], feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_K: f_K})

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
