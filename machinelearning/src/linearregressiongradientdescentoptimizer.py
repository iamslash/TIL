# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

def main():
    # set nodes
    X = [1, 2, 3]
    Y = [1, 2, 3]
    W = tf.Variable(5.0)
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # set cost function node
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)
    
    # launch node
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        print(step, sess.run(W))
        sess.run(train)
        
if __name__ == "__main__":
    main()
