# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

def main():
    # set train data
    ll_X = [[1, 2],
           [2, 3],
           [3, 1],
           [4, 3],
           [5, 3],
           [6, 2]]
    ll_Y = [[0],
           [0],
           [0],
           [1],
           [1],
           [1]]
    # set nodes
    t_X = tf.placeholder(tf.float32, shape=[None, 2])
    t_Y = tf.placeholder(tf.float32, shape=[None, 1])
    t_W = tf.Variable(tf.random_normal([2, 1]), name='W')
    t_b = tf.Variable(tf.random_normal([1]), name='b')
    t_H = tf.sigmoid(tf.matmul(t_X, t_W) + t_b)
    t_C = -tf.reduce_mean(t_Y * tf.log(t_H) + (1 - t_Y) * tf.log(1 - t_H))
    t_T = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(t_C)

    # accuracy computation
    t_pred = tf.cast(t_H > 0.5, dtype=tf.float32)
    t_accu = tf.reduce_mean(tf.cast(tf.equal(t_pred, t_Y), dtype=tf.float32))
    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(10001):
            f_cost, _ = sess.run([t_C, t_T], feed_dict={t_X: ll_X, t_Y: ll_Y})
            if n_step % 200 == 0:
                print(f'{n_step:10d} cost: {f_cost:10.7f}')

        # Accuracy report
        l_h, l_c, l_a = sess.run([t_H, t_pred, t_accu], feed_dict={t_X: ll_X, t_Y: ll_Y})
        print("\nHypothesis: ", l_h, "\nCorrect (Y): ", l_c, "\nAccuracy: ", l_a)
              
if __name__ == "__main__":
    main()