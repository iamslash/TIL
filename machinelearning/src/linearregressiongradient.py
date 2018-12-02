import tensorflow as tf
tf.set_random_seed(111)

def main():
    l_X = [1, 2, 3]
    l_Y = [1, 2, 3]
    t_W = tf.Variable(tf.random_normal([1]), name='W')
    t_X = tf.placeholder(tf.float32)
    t_Y = tf.placeholder(tf.float32)
    t_H = t_W * t_X
    t_C = tf.reduce_mean(tf.square(t_H - t_Y))

    f_learning_rate = 0.01
    t_G = tf.reduce_mean((t_W * t_X - t_Y) * t_X)  # gradient
    t_D = t_W - f_learning_rate * t_G              # descent
    t_U = t_W.assign(t_D)                          # update

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(101):
            _, f_C = sess.run([t_U, t_C], feed_dict={t_X: l_X, t_Y: l_Y})
            l_W = sess.run(t_W)
            print(f'{n_step:10d} {f_C:10.7f}', l_W)

if __name__ == '__main__':
    main()