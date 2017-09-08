# -*- coding: utf-8 -*-


def main():
    import tensorflow as tf
    tf.set_random_seed(777)

    # set nodes
    X = [1, 2, 3]
    Y = [1, 2, 3]
    W = tf.Variable(5.)
    hypothesis = W * X
    gradient = tf.reduce_mean((W * X - Y) * X) * 2
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # set cost function node
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)
    gvs = optimizer.compute_gradients(cost, [W])
    apply_gradients = optimizer.apply_gradients(gvs)

    # launch nodes
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)

if __name__ == "__main__":
    main()
