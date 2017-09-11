# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

def main():
    # set nodes 
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])
    hypothesis = X * W + b
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # train nodes
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                             feed_dict={X: [1, 2, 3],
                                                        Y: [1, 2, 3]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # test nodes
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

if __name__ == "__main__":
    main()
