# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def main():
    # set var

    f_learning_rate = 0.1

    # set data
    l_X = [[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]]
    l_Y = [[0],
           [1],
           [1],
           [0]]
    # set nodes
    l_X = np.array(l_X, dtype=np.float32)
    l_Y = np.array(l_Y, dtype=np.float32)
    t_X = tf.placeholder(tf.float32, [None, 2]) # ? x 2
    t_Y = tf.placeholder(tf.float32, [None, 1]) # ? x 1
    t_W = tf.Variable(tf.random_normal([2, 1]), name='W') # 2 x 1
    t_b = tf.Variable(tf.random_normal([1]), name='b')
    t_H = tf.sigmoid(tf.matmul(t_X, t_W) + t_b)
    t_C = -tf.reduce_mean(t_Y * tf.log(t_H) + (1 - t_Y) * tf.log(1 - t_H))
    t_T = tf.train.GradientDescentOptimizer(learning_rate=f_learning_rate).minimize(t_C)
    # set accuracy
    t_pred = tf.cast(t_H > 0.5, dtype=tf.float32)
    t_accu = tf.reduce_mean(tf.cast(tf.equal(t_pred, t_Y), dtype=tf.float32))

    # Launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for n_step in range(10001):
            sess.run(t_T, feed_dict={t_X: l_X, t_Y: l_Y})
            if n_step % 100 == 0:
                f_cost = sess.run(t_C, feed_dict={t_X: l_X, t_Y: l_Y})
                l_W = sess.run(t_W)
                print(f'{n_step:10d} cost: {f_cost:10.7f} W: \n', l_W)

        l_h, l_c, f_a = sess.run([t_H, t_pred, t_accu], feed_dict={t_X: l_X, t_Y: l_Y})
        print("\nHypothesis: ", l_h, "\nCorrect: ", l_c, "\nAccuracy: ", f_a)

if __name__ == "__main__":
    main()