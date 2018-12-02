# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(111)

def main():
    # set data
    l_X = [[73., 80., 75.],
           [93., 88., 93.],
           [89., 91., 90.],
           [96., 98., 100.],
           [73., 66., 70.]]
    l_Y = [[152.],
           [185.],
           [180.],
           [196.],
           [142.]]
    # set nodes
    t_X = tf.placeholder(tf.float32, shape=[None, 3])
    t_Y = tf.placeholder(tf.float32, shape=[None, 1])
    t_W = tf.Variable(tf.random_normal([3, 1]), name='W')
    t_b = tf.Variable(tf.random_normal([1]), name='B')
    t_H = tf.matmul(t_X, t_W) + t_b
    t_C = tf.reduce_mean(tf.square(t_H - t_Y))

    # set train node
    t_O = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    t_T = t_O.minimize(t_C)

    # train node
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(8001):
            f_C, l_H, _ = sess.run([t_C, t_H, t_T], feed_dict={t_X: l_X, t_Y: l_Y})
            if n_step % 10 == 0:
                print(f'{n_step:10d} cost: {f_C:10.7f} pred: ', l_H)

if __name__ == "__main__":
    main()