import tensorflow as tf

def main():
    tf.set_random_seed(111)
    l_X = [1, 2, 3]
    l_Y = [1, 2, 3]
    t_X = tf.placeholder(tf.float32)
    t_Y = tf.placeholder(tf.float32)
    t_W = tf.Variable(tf.random_normal([1]), name='W')
    t_b = tf.Variable(tf.random_normal([1]), name='b')
    t_H = l_X * t_W + t_b
    t_G = tf.reduce_mean((t_W * t_X - t_Y) * t_X) * 2
    t_C = tf.reduce_mean(tf.square(t_H - l_Y))
    t_O = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    t_T = t_O.minimize(t_C)
    t_GVS = t_O.compute_gradients(t_C, [t_W])
    t_apply_gradients = t_O.apply_gradients(t_GVS)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(100):
            l_fetch = sess.run([t_G, t_W, t_GVS], feed_dict={t_X: l_X, t_Y: l_Y})
            print("{:7d}".format(n_step), l_fetch)

if __name__ == "__main__":
    main()
