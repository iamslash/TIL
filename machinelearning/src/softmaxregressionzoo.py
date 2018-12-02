# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def main():
    # set data
    ll_XY = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
    ll_X  = ll_XY[:, 0:-1]
    ll_Y  = ll_XY[:, [-1]]

    # set nodes
    n_classes = 7  # 0 ~ 6

    t_X = tf.placeholder(tf.float32, [None, 16])
    t_Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
    t_Y_one_hot = tf.one_hot(t_Y, n_classes)  # one hot
    #print("one_hot", t_Y_one_hot)
    t_Y_one_hot = tf.reshape(t_Y_one_hot, [-1, n_classes])
    #print("reshape", Y_one_hot)
    t_W = tf.Variable(tf.random_normal([16, n_classes]), name='W')
    t_b = tf.Variable(tf.random_normal([n_classes]), name='b')

    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    t_logits = tf.matmul(t_X, t_W) + t_b
    t_H = tf.nn.softmax(t_logits)
    # Cross entropy cost/loss
    t_cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=t_logits, labels=t_Y_one_hot)
    t_cost   = tf.reduce_mean(t_cost_i)
    t_T = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(t_cost)
    t_pred = tf.argmax(t_H, 1)
    t_correct_prediction = tf.equal(t_pred, tf.argmax(t_Y_one_hot, 1))
    t_accuracy = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for n_step in range(2000):
            sess.run(t_T, feed_dict={t_X: ll_X, t_Y: ll_Y})
            if n_step % 100 == 0:
                f_cost, f_accu = sess.run([t_cost, t_accuracy], feed_dict={
                    t_X: ll_X, t_Y: ll_Y})
                print(f'{n_step:10d} cost: {f_cost:10.7f} accu: {f_accu:.2%}')

        # Let's see if we can predict
        l_pred = sess.run(t_pred, feed_dict={t_X: ll_X})
        # y_data: (N,1) = flatten => (N, ) matches pred.shape
        for p, y in zip(l_pred, ll_Y.flatten()):
            print(f'result: {p==int(y)} H(X): {p} Y: {int(y)}')
    
if __name__ == "__main__":
    main()