# -*- coding: utf-8 -*-
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # set variables
    tf.set_random_seed(777)

    # set data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # set hyper params
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    # set place holders
    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    # set W, L, b nodes
    # L1 ImgIn shape=(?, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    # conv -> (?, 28, 28, 32)
    # pool -> (?, 14, 14, 32)
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    # L2 ImgIn shape=(?, 14, 14, 32)
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    # conv -> (?, 14, 14, 64)
    # pool -> (?,  7,  7, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

    # Final FC 7x7x64 inputs -> 10 outputs
    W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                         initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L2_flat, W3) + b

    # set train node
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # launch nodes
    with tf.Session() as sess:
        print("started machine learning")
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch
                if i % 100 == 0:
                    print("\t", i, "avg_cost: ", avg_cost)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        print("ended machine learning")

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
            X: mnist.test.images, Y: mnist.test.labels}))

        # Get one and predict
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction: ", sess.run(
            tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
            
if __name__ == "__main__":
    main()
