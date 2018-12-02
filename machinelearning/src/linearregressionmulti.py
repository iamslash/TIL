import tensorflow as tf
tf.set_random_seed(111)

def main():
    l_X1 = [73., 93., 89., 96., 73.]
    l_X2 = [80., 88., 91., 98., 66.]
    l_X3 = [75., 93., 90., 100., 70.]
    l_Y  = [152., 185., 180., 196., 142.]
    t_X1 = tf.placeholder(tf.float32)
    t_X2 = tf.placeholder(tf.float32)
    t_X3 = tf.placeholder(tf.float32)
    t_Y  = tf.placeholder(tf.float32)
    t_W1 = tf.Variable(tf.random_normal([1]), name='W1')
    t_W2 = tf.Variable(tf.random_normal([1]), name='W2')
    t_W3 = tf.Variable(tf.random_normal([1]), name='W3')
    t_b  = tf.Variable(tf.random_normal([1]), name='b')
    t_H  = t_W1 * t_X1 + t_W2 * t_X2 + t_W3 * t_X3 + t_b
    print(t_H)

    t_C  = tf.reduce_mean(tf.square(t_H - t_Y))
    t_O  = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    t_T  = t_O.minimize(t_C)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(2001):
            f_C, f_H, _ = sess.run([t_C, t_H, t_T], 
                            feed_dict={t_X1: l_X1,
                            t_X2: l_X2,
                            t_X3: l_X3,
                            t_Y: l_Y})
            if n_step % 10 == 0:
                print(f'{n_step:10d} cost: {f_C:10.7f}', f_H)

if __name__ == '__main__':
    main()