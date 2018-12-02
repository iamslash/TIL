# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def main():
    # set data
    l_XY = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
    l_X = l_XY[:, 0:-1]
    l_Y = l_XY[:, [-1]]
    print(l_X.shape, l_X, len(l_X))
    print(l_Y.shape, l_Y)
    
    # set nodes
    t_X = tf.placeholder(tf.float32, shape=[None, 3])
    t_Y = tf.placeholder(tf.float32, shape=[None, 1])
    t_W = tf.Variable(tf.random_normal([3, 1]), name='W')
    t_b = tf.Variable(tf.random_normal([1]), name='b')
    t_H = tf.matmul(t_X, t_W) + t_b
    t_C = tf.reduce_mean(tf.square(t_H - t_Y))

    # set train node
    t_O = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    t_T = t_O.minimize(t_C)

    # train nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(2001):
            f_C, l_H, _ = sess.run(
                [t_C, t_H, t_T], feed_dict={t_X:l_X, t_Y:l_Y})
        if n_step % 10 == 0:
            print(f'{n_step:10d} cost: {f_C:10.7f} pred: ', l_H)

        # Ask my score
        ll_X = [[100, 70, 101]]
        print(ll_X, " will be ", sess.run(t_H, feed_dict={t_X: ll_X}))
        ll_X = [[60, 70, 110], [90, 100, 80]]
        print(ll_X, "will be ", sess.run(t_H, feed_dict={t_X: ll_X}))

if __name__ == "__main__":
    main()