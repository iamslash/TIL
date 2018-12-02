# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

def main():
    # set data
    ll_X = [[1, 2, 1, 1],
              [2, 1, 3, 2],
              [3, 1, 3, 4],
              [4, 1, 5, 5],
              [1, 7, 5, 5],
              [1, 2, 5, 6],
              [1, 6, 6, 6],
              [1, 7, 7, 7]]
    ll_Y = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0]]

    # set nodes
    t_X = tf.placeholder("float", [None, 4])
    t_Y = tf.placeholder("float", [None, 3])
    n_classes = 3
    t_W = tf.Variable(tf.random_normal([4, n_classes]), name='W')
    t_b = tf.Variable(tf.random_normal([n_classes]), name='b')
    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    t_H = tf.nn.softmax(tf.matmul(t_X, t_W) + t_b)
    t_C = tf.reduce_mean(-tf.reduce_sum(t_Y * tf.log(t_H), axis=1))
    t_T = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(t_C)

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for n_step in range(2001):
            sess.run(t_T, feed_dict={t_X: ll_X, t_Y: ll_Y})
            if n_step % 200 == 0:
                l_cost = sess.run(t_C, feed_dict={t_X: ll_X, t_Y: ll_Y})
                print(f'{n_step:10d}', l_cost)

        print('--------------')

        # Testing & One-hot encoding
        l_a = sess.run(t_H, feed_dict={t_X: [[1, 11, 7, 9]]})
        print(l_a, sess.run(tf.argmax(l_a, 1)))

        print('--------------')

        l_b = sess.run(t_H, feed_dict={t_X: [[1, 3, 4, 3]]})
        print(l_b, sess.run(tf.argmax(l_b, 1)))

        print('--------------')

        l_c = sess.run(t_H, feed_dict={t_X: [[1, 1, 0, 1]]})
        print(l_c, sess.run(tf.argmax(l_c, 1)))

        print('--------------')

        l_all = sess.run(t_H, feed_dict={
            t_X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
        print(l_all, sess.run(tf.argmax(l_all, 1)))
    
if __name__ == "__main__":
    main()