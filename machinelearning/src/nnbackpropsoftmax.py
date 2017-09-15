# -*- coding: utf-8 -*-

# Input: x
# Layer1: x * W + b
# Output layer = σ(Layer1)
#
# Loss_i = - y * log(σ(Layer1)) - (1 - y) * log(1 - σ(Layer1))
# Loss = tf.reduce_sum(Loss_i)
#
# We want to compute that
#
# dLoss/dW = ???
# dLoss/db = ???
#
# Network
#          p1     a1           l1 (y_pred)
# X -> (*) -> (+) -> (sigmoid) -> (loss)
#       ^      ^                 
#       |      |                 
#       W1     b1                


import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def sigma(x):
    # sigmoid function
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))

def sigma_prime(x):
    # derivative of the sigmoid function
    # σ'(x) = σ(x) * (1 - σ(x))
    return sigma(x) * (1. - sigma(x))  

def main():
    # set data
    xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
    X_data = xy[:, :-1]
    N = X_data.shape[0]
    y_data = xy[:, [-1]]
    # print("y has one of the following values")
    # print(np.unique(y_data))
    # print("Shape of X data: ", X_data.shape)
    # print("Shape of y data: ", y_data.shape)
    nb_classes = 7  # 0 ~ 6
    # set place holders
    X = tf.placeholder(tf.float32, [None, 16])
    y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
    # set nodes
    target = tf.one_hot(y, nb_classes)  # one hot
    target = tf.reshape(target, [-1, nb_classes])
    target = tf.cast(target, tf.float32)
    W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
    # set cost/loss node
    l1 = tf.matmul(X, W) + b
    y_pred = sigma(l1)
    loss_i = - target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
    loss = tf.reduce_sum(loss_i)
    # set back prop
    d_loss = (y_pred - target) / (y_pred * (1. - y_pred) + 1e-7)
    d_sigma = sigma_prime(l1)
    d_l1 = d_loss * d_sigma
    d_b = d_l1
    d_W = tf.matmul(tf.transpose(X), d_l1)
    # update network
    learning_rate = 0.01
    train = [
        tf.assign(W, W - learning_rate * d_W),
        tf.assign(b, b - learning_rate * tf.reduce_sum(d_b)),
    ]
    # set accuracy node
    prediction = tf.argmax(y_pred, 1)
    acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target, 1))
    acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

    # Launch graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(500):
            sess.run(train, feed_dict={X: X_data, y: y_data})

            if step % 10 == 0:
                # Within 300 steps, you should see an accuracy of 100%
                step_loss, acc = sess.run([loss, acct_res], feed_dict={
                    X: X_data, y: y_data})
                print("Step: {:5}\t Loss: {:10.5f}\t Acc: {:.2%}" .format(
                    step, step_loss, acc))

        # Let's see if we can predict
        pred = sess.run(prediction, feed_dict={X: X_data})
        for p, y in zip(pred, y_data):
            msg = "[{}]\t Prediction: {:d}\t True y: {:d}"
            print(msg.format(p == int(y[0]), p, int(y[0])))    

if __name__ == "__main__":
    main()

