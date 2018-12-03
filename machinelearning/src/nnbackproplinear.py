# -*- coding: utf-8 -*-
import tensorflow as tf
tf.set_random_seed(777)

# Network
#          p      l1 (y_pred)
# X -> (*) -> (+) -> (E)
#       ^      ^ 
#       |      | 
#       W      b
#
# ∂E/∂b =

def main():
    # set data
    l_X = [[73., 80., 75.],
           [93., 88., 93.],
           [89., 91., 90.],
           [96., 98., 100.],
           [73., 66., 70.]]
    l_Y = [[152.],
           [185.],
           [180.],
           [196.],
           [142.]]
    # set placeholder
    t_X = tf.placeholder(tf.float32, shape=[None, 3])
    t_Y = tf.placeholder(tf.float32, shape=[None, 1])
    # set nodes
    t_W = tf.Variable(tf.truncated_normal([3, 1]))
    t_b = tf.Variable(5.)
    t_H = tf.matmul(t_X, t_W) + t_b
    # set diff
    t_diff = (t_H - t_Y)
    # set back prop
    t_d_l1 = t_diff
    t_d_b = t_d_l1
    t_d_W = tf.matmul(tf.transpose(t_X), t_d_l1)
    # update network
    f_learning_rate = 1e-6
    l_t_step = [
        tf.assign(t_W, t_W - f_learning_rate * t_d_W),
        tf.assign(t_b, t_b - f_learning_rate * tf.reduce_mean(t_d_b))                  
    ]
    t_C = tf.reduce_mean(tf.square(t_Y - t_H))
    # launch nodes
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        l_step, f_cost = sess.run([l_t_step, t_C],
                              feed_dict={t_X: l_X, t_Y: l_Y})
        print(f'{i:10d} cost:{f_cost:10.7f} step:\n', l_step)
    print(sess.run(t_H, feed_dict={t_X: l_X}))

if __name__ == "__main__":
    main()