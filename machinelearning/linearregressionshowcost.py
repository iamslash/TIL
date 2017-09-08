# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

def main():
    # set node
    X = [1, 2, 3]
    Y = [1, 2, 3]
    W = tf.placeholder(tf.float32)
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # launch node
    sess = tf.Session()
    W_history = []
    cost_history = []
    for i in range(-30, 50):
        cur_W = i * 0.1 # learning rate
        cur_cost = sess.run(cost, feed_dict={W: cur_W})
        W_history.append(cur_W)
        cost_history.append(cur_cost)
    plt.plot(W_history, cost_history)
    plt.show()

if __name__ == "__main__":
    main()
