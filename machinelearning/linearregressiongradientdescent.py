# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

def main():
    # set nodes
    x_data = [1, 2, 3]
    y_data = [1, 2, 3]
    W = tf.Variable(tf.random_normal([1]), name='weight')
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # set gradient descent applied W node
    learning_rate = 0.1
    gradient = tf.reduce_mean((W * X - Y) * X)
    descent = W - learning_rate * gradient
    update = W.assign(descent)
    
    # launch node
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(21):
         sess.run(update, feed_dict = {X: x_data, Y: y_data})
         print(step, sess.run(cost, feed_dict = {X: x_data, Y: y_data}), sess.run(W))

if __name__ == "__main__":
    main()
