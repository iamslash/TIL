# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def main():
    # set data
    xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]

    # set nodes
    nb_classes = 7  # 0 ~ 6

    X = tf.placeholder(tf.float32, [None, 16])
    Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
    Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
    print("one_hot", Y_one_hot)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    print("reshape", Y_one_hot)
    W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.softmax(logits)
    # Cross entropy cost/loss
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2000):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={
                    X: x_data, Y: y_data})
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                    step, loss, acc))

        # Let's see if we can predict
        pred = sess.run(prediction, feed_dict={X: x_data})
        # y_data: (N,1) = flatten => (N, ) matches pred.shape
        for p, y in zip(pred, y_data.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    
if __name__ == "__main__":
    main()
