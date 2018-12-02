import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(111)

def main():
    l_X = [1, 2, 3]
    l_Y = [1, 2, 3]
    t_W = tf.placeholder(tf.float32)
    t_H = t_W * l_X
    t_C = tf.reduce_mean(tf.square(t_H - l_Y))

    with tf.Session() as sess:
        l_W_history = []
        l_C_history = []
        for i in range(-30, 50):
            f_W = i * 0.1
            f_C = sess.run(t_C, feed_dict={t_W: f_W})
            l_W_history.append(f_W)
            l_C_history.append(f_C)
        plt.plot(l_W_history, l_C_history)
        plt.show()

if __name__ == "__main__":
    main()