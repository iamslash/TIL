# -*- coding: utf-8 -*-

import tensorflow as tf

def main():
    tf.set_random_seed(777)
    x_train = [1, 2, 3]
    y_train = [1, 2, 3]

    # W, b, Hypothesis, cost
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = x_train * W + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    
    # minimize cost
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)
    
    # launch it
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))
            
if __name__ == "__main__":
    main()
