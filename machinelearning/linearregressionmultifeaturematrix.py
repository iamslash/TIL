# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

def main():
    # set data

    x_data = [[73., 80., 75.],
              [93., 88., 93.],
              [89., 91., 90.],
              [96., 98., 100.],
              [73., 66., 70.]]
    y_data = [[152.],
              [185.],
              [180.],
              [196.],
              [142.]]
    # set nodes
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = tf.matmul(X, W) + b
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # set train node
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # train node
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

if __name__ == "__main__":
    main()
