# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def main():
    # set variables
    tf.set_random_seed(777)

    # set data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # set hyper parameters
    learning_rate = 1e-3
    training_epochs = 15
    batch_size = 100

    # launch nodes
    with tf.Session() as sess:
        m1 = Model(sess, "m1")
        sess.run(tf.global_variables_initializer())
        print('Learning started!!!')
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    
        print('Learning ended!!!')
    
    
if __name__ == "__main__":
    main()
