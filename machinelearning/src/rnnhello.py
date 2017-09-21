# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def main():
    # set data
    idx2char = ['h', 'i', 'e', 'l', 'o']
    x_data = [[0, 1, 0, 2, 3, 4]]   # hihell
    x_one_hot = [[[1, 0, 0, 0, 0],  # h 0
                  [0, 1, 0, 0, 0],  # i 1
                  [1, 0, 0, 0, 0],  # h 0
                  [0, 0, 1, 0, 0],  # e 2
                  [0, 0, 0, 1, 0],  # l 3
                  [0, 0, 0, 1, 0]]] # l 3
    y_data = [[1, 0, 2, 3, 3, 4]]   # ihello

    # set variables
    num_classes     = 5
    input_dim       = 5  # input data is one hot
    hidden_size     = 5  # output from the LSTM. 5 to one-hot
    batch_size      = 1
    sequence_length = 6
    learning_rate   = 0.1

    # set placeholder
    X = tf.placeholder(
        tf.float32, [None, sequence_length, input_dim])
    Y = tf.placeholder(
        tf.int32, [None, sequence_length])

    # set RNN
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                        state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _state = tf.nn.dynamic_rnn(
        cell, X, initial_state=initial_state, dtype=tf.float32)

    # set FCNN
    x_for_fc = tf.reshape(outputs, [-1, hidden_size])
    # fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
    # fc_b = tf.get_variable("fc_b", [num_classes])
    # outputs = tf.matmul(X_for_fc, fc_w) + fc_b
    outputs = tf.contrib.layers.fully_connected(
        inputs=x_for_fc, num_outputs=num_classes, activation_fn=None)
    # reshape out for sequence_loss
    outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

    # set nodes
    weights = tf.ones([batch_size, sequence_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(
        logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    prediction = tf.argmax(outputs, axis=2)

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
            result = sess.run(prediction, feed_dict={X: x_one_hot})
            print(i, "loss: ", l, "pred: ", result, "true Y", y_data)
            #
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tpred str: ", ''.join(result_str))

if __name__ == "__main__":
    main()
