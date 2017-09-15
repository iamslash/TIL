# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

# Network
#          p      l1 (y_pred)
# X -> (*) -> (+) -> (E)
#       ^      ^ 
#       |      | 
#       W      b
#
# ∂E/∂b =

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
    # set placeholder
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    # set nodes
    W = tf.Variable(tf.truncated_normal([3, 1]))
    b = tf.Variable(5.)
    hypothesis = tf.matmul(X, W) + b
    # set diff
    diff = (hypothesis - Y)
    # set back prop
    d_l1 = diff
    d_b = d_l1
    d_w = tf.matmul(tf.transpose(X), d_l1)
    # update network
    learning_rate = 1e-6
    step = [
        tf.assign(W, W - learning_rate * d_w),
        tf.assign(b, b - learning_rate * tf.reduce_mean(d_b))                  
    ]
    RMSE = tf.reduce_mean(tf.square(Y - hypothesis))
    # launch nodes
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        print(i, sess.run([step, RMSE],
                              feed_dict={X: x_data, Y:y_data}))
    print(sess.run(hypothesis, feed_dict={X: x_data}))

if __name__ == "__main__":
    main()
