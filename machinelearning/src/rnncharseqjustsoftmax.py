# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def main():
    # set characters
    sample = " if you want you"
    idx2char = list(set(sample))
    char2idx = {c: i for i, c in enumerate(idx2char)}

    # set hyper params
    dic_size = len(char2idx) # RNN input size (one hot size)
    hidden_size = len(char2idx) # RNN output size
    num_classes = len(char2idx) # final output size (RNN or softmax)
    batch_size = 1 # sample data count
    sequence_length = len(sample) - 1 # number of LSTM rollings
    learning_rate = 0.1

    # set data
    sample_idx = [char2idx[c] for c in sample]
    x_data = [sample_idx[:-1]]
    y_data = [sample_idx[1:]]

    # set placeholder
    X = tf.placeholder(tf.int32, [None, sequence_length])
    Y = tf.placeholder(tf.int32, [None, sequence_length])

    # flatten the data
    x_one_hot = tf.one_hot(X, num_classes)
    X_for_softmax = tf.reshape(x_one_hot, [-1, hidden_size])

    # set softmax layer
    softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
    softmax_b = tf.get_variable("softmax_b", [num_classes])
    outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

    # expand the data
    outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
    weights = tf.ones([batch_size, sequence_length])

    # set nodes
    sequence_loss = tf.contrib.seq2seq.sequence_loss(
        logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    prediction = tf.argmax(outputs, axis=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
            result = sess.run(prediction, feed_dict={X: x_data})

            # print
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print(i, "loss: ", l, "Prediction: ", ''.join(result_str))

if __name__ == "__main__":
    main()
