import tensorflow as tf

def main():
    tf.set_random_seed(111)
    l_X = [1, 2, 3]
    l_Y = [1, 2, 3]
    t_W = tf.Variable(tf.random_normal([1]), name='W')
    t_b = tf.Variable(tf.random_normal([1]), name='b')
    t_H = l_X * t_W + t_b
    t_C = tf.reduce_mean(tf.square(t_H - l_Y))
    t_O = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    t_T = t_O.minimize(t_C)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(10000):
            sess.run(t_T)
            if n_step % 20 == 0:
                f_cost = sess.run(t_C)
                l_W = sess.run(t_W)
                l_b = sess.run(t_b)

                #print("step = {:7d} loss = {:5.3f} W = {:5.3f} b = {:5.3f}".format(step, f_cost, f_W, f_b) )
                print("{:7d}".format(n_step), "{:10.7f}".format(f_cost), l_W, l_b)

if __name__ == "__main__":
    main()