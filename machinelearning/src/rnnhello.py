# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

def main():
    # set data
    idx2char = ['h', 'i', 'e', 'l', 'o']
    l_X = [[0, 1, 0, 2, 3, 4]]   # hihell
    l_X_one_hot = [[[1, 0, 0, 0, 0],  # h 0
                   [0, 1, 0, 0, 0],  # i 1
                   [1, 0, 0, 0, 0],  # h 0
                   [0, 0, 1, 0, 0],  # e 2
                   [0, 0, 0, 1, 0],  # l 3
                   [0, 0, 0, 1, 0]]] # l 3
    l_Y = [[1, 0, 2, 3, 3, 4]]   # ihello

    # set variables
    n_class_cnt     = 5
    n_input_dim     = 5  # input data is one hot
    n_hidden_size   = 5  # output from the LSTM. 5 to one-hot
    n_batch_size    = 1
    n_seq_len       = 6
    f_learn_rate    = 0.1

    # set placeholder
    t_X = tf.placeholder(tf.float32, [None, n_seq_len, n_input_dim])
    t_Y = tf.placeholder(tf.int32, [None, n_seq_len])

    # set RNN
    cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(n_batch_size, tf.float32)
    t_outputs, _ = tf.nn.dynamic_rnn(cell, t_X, initial_state=initial_state, dtype=tf.float32)

    # set FCNN
    t_X_for_fc = tf.reshape(t_outputs, [-1, n_hidden_size])
    # fc_w = tf.get_variable("fc_w", [n_hidden_size, n_class_cnt])
    # fc_b = tf.get_variable("fc_b", [n_class_cnt])
    # t_outputs = tf.matmul(X_for_fc, fc_w) + fc_b
    t_outputs = tf.contrib.layers.fully_connected(inputs=t_X_for_fc, num_outputs=n_class_cnt, activation_fn=None)
    # reshape out for sequence_loss
    t_outputs = tf.reshape(t_outputs, [n_batch_size, n_seq_len, n_class_cnt])

    # set nodes
    weights = tf.ones([n_batch_size, n_seq_len])
    t_seq_loss = tf.contrib.seq2seq.sequence_loss(logits=t_outputs, targets=t_Y, weights=weights)
    t_C = tf.reduce_mean(t_seq_loss)
    t_T = tf.train.AdamOptimizer(learning_rate=f_learn_rate).minimize(t_C)
    t_pred = tf.argmax(t_outputs, axis=2)

    # launch nodes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            f_cost, _ = sess.run([t_C, t_T], feed_dict={t_X: l_X_one_hot, t_Y: l_Y})
            l_pred = sess.run(t_pred, feed_dict={t_X: l_X_one_hot})
            pred_str = ''.join([idx2char[c] for c in np.squeeze(l_pred)])
            true_str = ''.join([idx2char[c] for c in np.squeeze(l_Y)])
            print(f'{i:10d}, pred: {pred_str}, true: {true_str}')
        #print(sess.run(weights))

if __name__ == "__main__":
    main()
