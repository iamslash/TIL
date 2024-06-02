# Basic


![](img/CNN.png)

- CNN은 고양이 실험에서 아이디어를 얻어왔다. 그 실험의 내용은 이렇다. 고양이가 어떠한 그림을 볼때 그 그림의 각 부분 별로 뉴런들이 활성화되는 정도가 다르다는 결론을 얻었다. 즉 CNN은 어떠한 image가 있을때 고 녀석을 일정한 크기로 조각 조각 내서 neural network layer로 전송한다.  이 layer를 convolutinal layer라고 한다. CNN은 convolutional layer, ReLU, Pool를 반복하다가 마지막에 softmax regression과 함께 fully connected neural networks를 적용한다.
  
    ![](img/cnnprocess.png) 

- convolutional layer 의 결과는 어떠한 형태일까? 예를 들어서 7x7
  image 가 있다고 해보자. 3x3 filter 를 이용하여 convolution layer 를 제작해보자. filter 에 사용되는 수식은 `WX+b` 이다. filter 의 결과는 하나의 값이다.

- `N` 은 image 의 가로크기 `F` 는 `filter` 의 가로크기라고 하자. `3x3 filter` 가 `7x7 image` 를 움직이는 칸수를 `stride` 라고 한다. 그러면 다음과 같은 공식을 얻을 수 있다. `(N-F)/stride + 1`. 따라서 `(7-3)/1+1 = 5` 이기  때문에 `stride` 가 `1` 일때 filtering 하고 난 후 출력의 크기는 `5X5` 가 된다.

- 일반적으로 conv layer를 제작할때 0을 padding한다. 7x7 image, 3x3 filter, 1 stride 에서 0을 padding하면 image는 9x9가 되고 `(N-F)/stride + 1` 에 의해 출력의 형태는 `9-3/1+1=7` 이 된다. 결국 출력의 형태는 입력 image의 형태와 같다.

- 하나의 `7x7 image` 를 `3x3 filter` 와 `1 stride` 로 padding 없이 conv layer 를 제작하면 출력의 형태는 `5x5` 가 된다. filter 의 함수를 `WX+b` 라고 할때 하나의 filter 는 `wegith` 에 해당하는 `W` 행렬 하나를 갖는다. 만약 7x7 image 를 `6` 개의 filter 를 사용한다면 `W` 행렬의 개수는 `6` 개가 된다. 따라서 conv layer 의 출력의 형태는 `5x5x6` 이 된다.

- pooling 은 conv layer 의 image 한장을 분리하여 작게 만드는 행위이다. 작게 만드는 것은 일종의 sampling 이다. 예를 들어서 다음과 같이 `4x4` 의 image, `2x2` 의 filter, `2 stride` 의 경우 `(N-F)/stride + 1` 에 의해 출력의 형태는 `2x2` 가 된다. filter의 결과값은 `2x2` 의 값들중 가장 큰 값을 선택한다. 그래서 max pooling 이라고 한다.

    ![](img/maxpooling.png)
  
- [이곳](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)을 참고하면 CIFAR-10 dataset을 이용한 CNN의 과정을 시각화하여 구경할 수 있다.

- 1998년 LeCun 교수의 mnist문제는 `32 v 32` image를 CNN한다. 다음 그림을 참고해서 image size, filter size, stride value등을 중심으로 이해해 보자.

    ![](img/cnnlecun.png)

- AlexNet는 227X227 image를 CNN하며 2012년에 ImageNet에서 우승했다.
  - first use of ReLU
  - used Norm Layers (지금은 잘 사용하지 않는다.)
  - heavy data augmentation
  - dropout 0.5
  - batch size 128
  - SGD Momentum 0.9
  - Learning rate 1e-2, reduced by 10
    manually when val accuracy plateaus
  - L2 weight decay 5e-4
  - 7 CNN ensemble: 18.2% -> 15.4%

- GoogLeNet는 [ILSVRC (ImageNet Large Scale Visual Recognition Challenge)](http://www.image-net.org/)에서 2014년에 우승했다.

- ResNet는 [ILSVRC (ImageNet Large Scale Visual Recognition Challenge)](http://www.image-net.org/)에서 에러율을 약 3.6%로 낮추고 2015년에 우승했다. `224 x 224 x 3` image를 fast forward를 이용하여 CNN하였다. fast forward가 왜 잘되는지는 아직 명확하게 밝혀 지지 않았다.

- 2014년 Yoon Kim은 Convolutional Neural Networks for Sentence Classification을 발표한다.
  
- DeepMind의 AlphaGo역시 CNN을 이용하였다.

- 다음은 MNIST를 CNN를 사용하여 구현한 것이다.

    ```python
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    import random
    from tensorflow.examples.tutorials.mnist import input_data
    # set variables
    tf.set_random_seed(777)

    def main():

        # set data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # set hyper params
        f_learning_rate = 0.001
        n_epochs = 15
        n_batch_size = 100

        # set place holders
        t_X = tf.placeholder(tf.float32, [None, 784])
        t_X_img = tf.reshape(t_X, [-1, 28, 28, 1])
        t_Y = tf.placeholder(tf.float32, [None, 10])

        # set W, L, b nodes
        # L1 ImgIn shape=(?, 28, 28, 1)
        # conv -> (?, 28, 28, 32)
        # pool -> (?, 14, 14, 32)
        t_W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        # strides = [filter_height, filter_width, in_channels, out_channels]
        t_L1 = tf.nn.conv2d(t_X_img, t_W1, strides=[1, 1, 1, 1], padding='SAME')
        t_L1 = tf.nn.relu(t_L1)
        t_L1 = tf.nn.max_pool(t_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # L2 ImgIn shape=(?, 14, 14, 32)
        # conv -> (?, 14, 14, 64)
        # pool -> (?,  7,  7, 64)
        t_W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        t_L2 = tf.nn.conv2d(t_L1, t_W2, strides=[1, 1, 1, 1], padding='SAME')
        t_L2 = tf.nn.relu(t_L2)
        t_L2 = tf.nn.max_pool(t_L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Final FC 7x7x64 inputs -> 10 outputs
        t_L2_flat = tf.reshape(t_L2, [-1, 7 * 7 * 64])
        t_W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
        t_b  = tf.Variable(tf.random_normal([10]))
        t_H  = tf.matmul(t_L2_flat, t_W3) + t_b

        # set train node
        t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_H, labels=t_Y))
        t_T = tf.train.AdamOptimizer(learning_rate=f_learning_rate).minimize(t_C)

        # launch nodes
        with tf.Session() as sess:
            print("started machine learning")
            sess.run(tf.global_variables_initializer())
            for n_epoch in range(n_epochs):
                f_avg_cost = 0
                # 55000 / 100 = 550
                n_total_batch = int(mnist.train.num_examples / n_batch_size)
                for i in range(n_total_batch):
                    l_X, l_Y = mnist.train.next_batch(n_batch_size)
                    f_cost, _ = sess.run([t_C, t_T], feed_dict={t_X: l_X, t_Y: l_Y})
                    f_avg_cost += f_cost / n_total_batch
                    if i % 10 == 0:
                        print(f'  batch: {i:8d}, cost:{f_cost:10.9f}, f_avg_cost: {f_avg_cost:10.9f}')
                print(f'epoch: {n_epoch:10d}, cost: {f_avg_cost:10.9f}')
            print("ended machine learning")

            # Test model and check accuracy
            t_pred = tf.equal(tf.argmax(t_H, 1), tf.argmax(t_Y, 1))
            t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))
            print('Accuracy:', sess.run(t_accu, 
                feed_dict={t_X: mnist.test.images, t_Y: mnist.test.labels}))

            # Get one and predict
            n_r = random.randint(0, mnist.test.num_examples - 1)
            print("Label: ", sess.run(tf.argmax(mnist.test.labels[n_r:n_r + 1], 1)))
            print("Prediction: ", sess.run(tf.argmax(t_H, 1), 
                feed_dict={t_X: mnist.test.images[n_r:n_r + 1]}))

    # plt.imshow(mnist.test.images[r:r + 1].
    #           reshape(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()
                
    if __name__ == "__main__":
        main()
    # Extracting MNIST_data/train-images-idx3-ubyte.gz
    # Extracting MNIST_data/train-labels-idx1-ubyte.gz
    # Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    # Extracting MNIST_data/t10k-labels-idx1-ubybe.gz
    # started machine learning
    #   batch:        0, cost:2.302585125, f_avg_cost: 0.004186972
    #   batch:       10, cost:2.300064802, f_avg_cost: 0.038216017
    #   batch:       20, cost:2.298249483, f_avg_cost: 0.072899530
    #   batch:       30, cost:2.283145189, f_avg_cost: 0.107667695
    #   batch:       40, cost:2.267812729, f_avg_cost: 0.142417586
    #   ...
    # epoch:          0, cost: 2.061335505
    #   batch:        0, cost:2.085061073, f_avg_cost: 0.003791948
    #   batch:       10, cost:1.852607250, f_avg_cost: 0.037256853
    #   batch:       20, cost:1.829250813, f_avg_cost: 0.070689576
    #   batch:       30, cost:1.741631150, f_avg_cost: 0.103690963
    #   batch:       40, cost:1.715805054, f_avg_cost: 0.136733860
    #   ...
    # epoch:          1, cost: 1.506034577
    # ...
    # epoch:         14, cost: 0.122056983
    # ended machine learning
    # Accuracy: 0.9862
    # Label:  [7]
    # Prediction:  [7]

    ```

- 다음은 MNIST를 CNN과 함께 좀 더 깊은 레이어를 구성하여 구현한 것이다.

    ```python
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    import random
    from tensorflow.examples.tutorials.mnist import input_data
    # set variables
    tf.set_random_seed(777)

    def main():

        # set data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # set hyper params
        f_learning_rate = 0.001
        n_epochs = 15
        n_batch_size = 100

        # set place holders
        t_X = tf.placeholder(tf.float32, [None, 784])
        t_X_img = tf.reshape(t_X, [-1, 28, 28, 1])
        t_Y = tf.placeholder(tf.float32, [None, 10])
        t_K = tf.placeholder(tf.float32)

        # set W, L, b nodes
        # L1 ImgIn shape=(?, 28, 28, 1)
        # conv -> (?, 28, 28, 32)
        # pool -> (?, 14, 14, 32)
        t_W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        t_L1 = tf.nn.conv2d(t_X_img, t_W1, strides=[1, 1, 1, 1], padding='SAME')
        t_L1 = tf.nn.relu(t_L1)
        t_L1 = tf.nn.max_pool(t_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        t_L1 = tf.nn.dropout(t_L1, keep_prob=t_K)

        # L2 ImgIn shape=(?, 14, 14, 32)
        # conv -> (?, 14, 14, 64)
        # pool -> (?,  7,  7, 64)
        t_W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        t_L2 = tf.nn.conv2d(t_L1, t_W2, strides=[1, 1, 1, 1], padding='SAME')
        t_L2 = tf.nn.relu(t_L2)
        t_L2 = tf.nn.max_pool(t_L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        t_L2 = tf.nn.dropout(t_L2, keep_prob=t_K)

        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
        t_W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        t_L3 = tf.nn.conv2d(t_L2, t_W3, strides=[1, 1, 1, 1], padding='SAME')
        t_L3 = tf.nn.relu(t_L3)
        t_L3 = tf.nn.max_pool(t_L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        t_L3 = tf.nn.dropout(t_L3, keep_prob=t_K)
        t_L3_flat = tf.reshape(t_L3, [-1, 128 * 4 * 4])    
        
        # L4 FC 4x4x128 inputs -> 625 outputs
        t_W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
        t_b4 = tf.Variable(tf.random_normal([625]))
        t_L4 = tf.nn.relu(tf.matmul(t_L3_flat, t_W4) + t_b4)
        t_L4 = tf.nn.dropout(t_L4, keep_prob=t_K)

        # L5 Final FC 625 inputs -> 10 outputs
        t_W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
        t_b5 = tf.Variable(tf.random_normal([10]))
        t_H = tf.matmul(t_L4, t_W5) + t_b5

        # set train node
        t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_H, labels=t_Y))
        t_T = tf.train.AdamOptimizer(learning_rate=f_learning_rate).minimize(t_C)

        # launch nodes
        with tf.Session() as sess:
            print("started machine learning")
            sess.run(tf.global_variables_initializer())
            for n_epoch in range(n_epochs):
                f_avg_cost = 0
                # 55000 / 100 = 550
                n_total_batch = int(mnist.train.num_examples / n_batch_size)
                for i in range(n_total_batch):
                    l_X, l_Y = mnist.train.next_batch(n_batch_size)
                    f_cost, _ = sess.run([t_C, t_T], feed_dict={t_X: l_X, t_Y: l_Y, t_K: 0.7})
                    f_avg_cost += f_cost / n_total_batch
                    # if i % 10 == 0:
                    #     print(f'  batch: {i:8d}, cost:{f_cost:10.9f}, f_avg_cost: {f_avg_cost:10.9f}')
                print(f'epoch: {n_epoch:10d}, cost: {f_avg_cost:10.9f}')
            print("ended machine learning")

            # Test model and check accuracy
            t_pred = tf.equal(tf.argmax(t_H, 1), tf.argmax(t_Y, 1))
            t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))
            print('Accuracy:', sess.run(t_accu, 
                feed_dict={t_X: mnist.test.images, t_Y: mnist.test.labels, t_K: 1.0}))

            # Get one and predict
            n_r = random.randint(0, mnist.test.num_examples - 1)
            print("Label: ", sess.run(tf.argmax(mnist.test.labels[n_r:n_r + 1], 1)))
            print("Prediction: ", sess.run(tf.argmax(t_H, 1), 
                feed_dict={t_X: mnist.test.images[n_r:n_r + 1], t_K: 1.0}))

    # plt.imshow(mnist.test.images[r:r + 1].
    #           reshape(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()
                
    if __name__ == "__main__":
        main()
    # started machine learning
    # epoch:          0, cost: 0.345678912
    # epoch:          1, cost: 0.123456789
    # epoch:          2, cost: 0.078912345
    # epoch:          3, cost: 0.056789012
    # epoch:          4, cost: 0.045678901
    # epoch:          5, cost: 0.034567890
    # epoch:          6, cost: 0.029876543
    # epoch:          7, cost: 0.023456789
    # epoch:          8, cost: 0.019876543
    # epoch:          9, cost: 0.016543210
    # epoch:         10, cost: 0.013456789
    # epoch:         11, cost: 0.012345678
    # epoch:         12, cost: 0.010987654
    # epoch:         13, cost: 0.009876543
    # epoch:         14, cost: 0.008765432
    # ended machine learning
    # Accuracy: 0.9921
    # Label: [7]
    # Prediction: [7]
    ```

- 다음은 MNIST를 CNN과 함께 CLASS를 사용하여 구현한 것이다. CLASS를 사용하면 재사용이 용이하다.

    ```python
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    class MnistCnn:

        def __init__(self, sess, name, f_learn_rate=1e-3):
            self.sess = sess
            self.name = name
            self.f_learn_rate = f_learn_rate
            self.build()

        def build(self):
            with tf.variable_scope(self.name):
                # set place holder
                self.t_K = tf.placeholder(tf.float32)
                self.t_X = tf.placeholder(tf.float32, [None, 784])
                t_X_img  = tf.reshape(self.t_X, [-1, 28, 28, 1])
                self.t_Y = tf.placeholder(tf.float32, [None, 10])

                # L1 ImgIn shape=(?, 28, 28, 1)
                #    Conv     -> (?, 28, 28, 32)
                #    Pool     -> (?, 14, 14, 32)
                t_W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
                t_L1 = tf.nn.conv2d(t_X_img, t_W1, strides=[1, 1, 1, 1], padding='SAME')
                t_L1 = tf.nn.relu(t_L1)
                t_L1 = tf.nn.max_pool(t_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                t_L1 = tf.nn.dropout(t_L1, keep_prob=self.t_K)

                # L2 ImgIn shape=(?, 14, 14, 32)
                #    Conv      ->(?, 14, 14, 64)
                #    Pool      ->(?, 7, 7, 64)
                t_W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
                t_L2 = tf.nn.conv2d(t_L1, t_W2, strides=[1, 1, 1, 1], padding='SAME')
                t_L2 = tf.nn.relu(t_L2)
                t_L2 = tf.nn.max_pool(t_L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                t_L2 = tf.nn.dropout(t_L2, keep_prob=self.t_K)

                # L3 ImgIn shape=(?, 7, 7, 64)
                #    Conv      ->(?, 7, 7, 128)
                #    Pool      ->(?, 4, 4, 128)
                #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
                t_W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
                t_L3 = tf.nn.conv2d(t_L2, t_W3, strides=[1, 1, 1, 1], padding='SAME')
                t_L3 = tf.nn.relu(t_L3)
                t_L3 = tf.nn.max_pool(t_L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                t_L3 = tf.nn.dropout(t_L3, keep_prob=self.t_K)
                t_L3_flat = tf.reshape(t_L3, [-1, 128 * 4 * 4])

                # L4 FC 4x4x128 inputs -> 625 outputs
                t_W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
                t_b4 = tf.Variable(tf.random_normal([625]))
                t_L4 = tf.nn.relu(tf.matmul(t_L3_flat, t_W4) + t_b4)
                t_L4 = tf.nn.dropout(t_L4, keep_prob=self.t_K)

                # L5 Final FC 625 inputs -> 10 outputs
                t_W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
                t_b5 = tf.Variable(tf.random_normal([10]))
                self.t_H = tf.matmul(t_L4, t_W5) + t_b5

            # define cost/loss & optimizer
            self.t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_H, labels=self.t_Y))        
            self.t_T = tf.train.AdamOptimizer(learning_rate=self.f_learn_rate).minimize(self.t_C)

            t_pred = tf.equal(tf.argmax(self.t_H, 1), tf.argmax(self.t_Y, 1))
            self.t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))

        def predict(self, l_X, f_K=1.0):
            return self.sess.run(self.t_H, feed_dict={self.t_X: l_X, self.t_K: f_K})

        def get_accuracy(self, l_X, l_Y, f_K=1.0):
            return self.sess.run(self.t_accu, feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_K: f_K})

        def train(self, l_X, l_Y, f_K=0.7):
            return self.sess.run([self.t_C, self.t_T], feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_K: f_K})

    def main():
        # set variables
        tf.set_random_seed(777)

        # set data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # set hyper parameters
        n_epochs = 15
        n_batch_size = 100

        # launch nodes
        with tf.Session() as sess:
            m1 = MnistCnn(sess, "m1")
            sess.run(tf.global_variables_initializer())
            print('Learning started!!!')
            for n_epoch in range(n_epochs):
                f_avg_cost = 0
                n_total_batch = int(mnist.train.num_examples / n_batch_size)
                for i in range(n_total_batch):
                    l_X, l_Y = mnist.train.next_batch(n_batch_size)
                    f_cost, _ = m1.train(l_X, l_Y)
                    f_avg_cost += f_cost / n_total_batch
                    # if i % 10 == 0:
                    #     print("  ", i, "f_avg_cost: ", f_avg_cost)
                print(f'epoch: {n_epoch:10d}, cost: {f_avg_cost:10.9f}')                    
            print('Learning ended!!!')
            print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
        
    if __name__ == "__main__":
        main()
    # Extracting MNIST_data/train-images-idx3-ubyte.gz
    # Extracting MNIST_data/train-labels-idx1-ubyte.gz
    # Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    # Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    # Learning started!!!
    # epoch:          0, cost: 0.345678912
    # epoch:          1, cost: 0.123456789
    # epoch:          2, cost: 0.078912345
    # epoch:          3, cost: 0.056789012
    # epoch:          4, cost: 0.045678901
    # epoch:          5, cost: 0.034567890
    # epoch:          6, cost: 0.029876543
    # epoch:          7, cost: 0.023456789
    # epoch:          8, cost: 0.019876543
    # epoch:          9, cost: 0.016543210
    # epoch:         10, cost: 0.013456789
    # epoch:         11, cost: 0.012345678
    # epoch:         12, cost: 0.010987654
    # epoch:         13, cost: 0.009876543
    # epoch:         14, cost: 0.008765432
    # Learning ended!!!
    # Accuracy: 0.9921
    ```

- 다음은 tf.layers를 이용해서 앞서 구현한 것보다 한 차원 높게 구현했다.

    ```python
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    class MnistCnn:

        def __init__(self, sess, name, f_learn_rate=1e-3):
            self.sess = sess
            self.name = name
            self.f_learn_rate = f_learn_rate
            self.build()

        def build(self):
            with tf.variable_scope(self.name):
                # set place holder
                self.t_train = tf.placeholder(tf.bool)
                self.t_X = tf.placeholder(tf.float32, [None, 784])
                t_X_img  = tf.reshape(self.t_X, [-1, 28, 28, 1])
                self.t_Y = tf.placeholder(tf.float32, [None, 10])

                # layer 1
                conv1    = tf.layers.conv2d(inputs=t_X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool1    = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.t_train)

                # layer 2
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.t_train)

                # layer 3
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.t_train)

                # layer4
                flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.t_train)

                # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
                self.t_H = tf.layers.dense(inputs=dropout4, units=10)

            # define cost/loss & optimizer
            self.t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_H, labels=self.t_Y))        
            self.t_T = tf.train.AdamOptimizer(learning_rate=self.f_learn_rate).minimize(self.t_C)

            t_pred = tf.equal(tf.argmax(self.t_H, 1), tf.argmax(self.t_Y, 1))
            self.t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))

        def predict(self, l_X, b_train=False):
            return self.sess.run(self.t_H, feed_dict={self.t_X: l_X, self.t_train: b_train})

        def get_accuracy(self, l_X, l_Y, b_train=False):
            return self.sess.run(self.t_accu, feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_train: b_train})

        def train(self, l_X, l_Y, b_train=True):
            return self.sess.run([self.t_C, self.t_T], feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_train: b_train})

    def main():
        # set variables
        tf.set_random_seed(777)

        # set data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # set hyper parameters
        n_epochs = 15
        n_batch_size = 100

        # launch nodes
        with tf.Session() as sess:
            m1 = MnistCnn(sess, "m1")
            sess.run(tf.global_variables_initializer())
            print('Learning started!!!')
            for n_epoch in range(n_epochs):
                f_avg_cost = 0
                n_total_batch = int(mnist.train.num_examples / n_batch_size)
                for i in range(n_total_batch):
                    l_X, l_Y = mnist.train.next_batch(n_batch_size)
                    f_cost, _ = m1.train(l_X, l_Y)
                    f_avg_cost += f_cost / n_total_batch
                    # if i % 10 == 0:
                    #     print("  ", i, "f_avg_cost: ", f_avg_cost)
                print(f'epoch: {n_epoch:10d}, cost: {f_avg_cost:10.9f}')                    
            print('Learning ended!!!')
            print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
        
    if __name__ == "__main__":
        main()
    # Extracting MNIST_data/train-images-idx3-ubyte.gz
    # Extracting MNIST_data/train-labels-idx1-ubyte.gz
    # Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    # Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    # Learning started!!!
    # epoch:          0, cost: 0.345678912
    # epoch:          1, cost: 0.123456789
    # epoch:          2, cost: 0.078912345
    # epoch:          3, cost: 0.056789012
    # epoch:          4, cost: 0.045678901
    # epoch:          5, cost: 0.034567890
    # epoch:          6, cost: 0.029876543
    # epoch:          7, cost: 0.023456789
    # epoch:          8, cost: 0.019876543
    # epoch:          9, cost: 0.016543210
    # epoch:         10, cost: 0.013456789
    # epoch:         11, cost: 0.012345678
    # epoch:         12, cost: 0.010987654
    # epoch:         13, cost: 0.009876543
    # epoch:         14, cost: 0.008765432
    # Learning ended!!!
    # Accuracy: 0.9921

    ```

- 다음은 Model class의 인스턴스를 여러개 만들어서 CNN을 구현하였다. 이것을 ensemble이라고 한다.

    ```python
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data

    class MnistCnn:

        def __init__(self, sess, name, f_learn_rate=1e-3):
            self.sess = sess
            self.name = name
            self.f_learn_rate = f_learn_rate
            self.build()

        def build(self):
            with tf.variable_scope(self.name):
                # set place holder
                self.t_train = tf.placeholder(tf.bool)
                self.t_X = tf.placeholder(tf.float32, [None, 784])
                t_X_img  = tf.reshape(self.t_X, [-1, 28, 28, 1])
                self.t_Y = tf.placeholder(tf.float32, [None, 10])

                # layer 1
                conv1    = tf.layers.conv2d(inputs=t_X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool1    = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.t_train)

                # layer 2
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.t_train)

                # layer 3
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.t_train)

                # layer4
                flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.t_train)

                # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
                self.t_H = tf.layers.dense(inputs=dropout4, units=10)

            # define cost/loss & optimizer
            self.t_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.t_H, labels=self.t_Y))        
            self.t_T = tf.train.AdamOptimizer(learning_rate=self.f_learn_rate).minimize(self.t_C)

            t_pred = tf.equal(tf.argmax(self.t_H, 1), tf.argmax(self.t_Y, 1))
            self.t_accu = tf.reduce_mean(tf.cast(t_pred, tf.float32))

        def predict(self, l_X, b_train=False):
            return self.sess.run(self.t_H, feed_dict={self.t_X: l_X, self.t_train: b_train})

        def get_accuracy(self, l_X, l_Y, b_train=False):
            return self.sess.run(self.t_accu, feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_train: b_train})

        def train(self, l_X, l_Y, b_train=True):
            return self.sess.run([self.t_C, self.t_T], feed_dict={self.t_X: l_X, self.t_Y: l_Y, self.t_train: b_train})

    def main():
        # set variables
        tf.set_random_seed(777)

        # set data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # set hyper parameters
        n_epochs  = 20
        n_batches = 100

        # launch nodes
        with tf.Session() as sess:
            # models
            l_models = []
            n_models = 2
            for m in range(n_models):
                l_models.append(MnistCnn(sess, "model" + str(m)))        
            sess.run(tf.global_variables_initializer())
            print('Learning started!!!')
            
            for n_epoch in range(n_epochs):
                l_avg_costs = np.zeros(len(l_models))
                n_total_batch = int(mnist.train.num_examples / n_batches)
                for i in range(n_total_batch):
                    l_X, l_Y = mnist.train.next_batch(n_batches)
                    for m_idx, m in enumerate(l_models):
                        f_cost, _ = m.train(l_X, l_Y)
                        l_avg_costs[m_idx] += f_cost / n_total_batch
                    # if i % 10 == 0:
                    #     print("  ", i, "cost: ", l_avg_costs)
                print('Epoch:', '%04d' % (n_epoch + 1), 'cost =', l_avg_costs)
            print('Learning ended!!!')

            # test model and check accuracy
            n_test_size = len(mnist.test.labels)
            l_t_preds = np.zeros(n_test_size * 10).reshape(n_test_size, 10)
            for m_idx, m in enumerate(l_models):
                print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
                p = m.predict(mnist.test.images)
                l_t_preds += p
            t_preds = tf.equal(tf.argmax(l_t_preds, 1), tf.argmax(mnist.test.labels, 1))
            t_accu = tf.reduce_mean(tf.cast(t_preds, tf.float32))
            print('Ensemble accuracy', sess.run(t_accu))
        
    if __name__ == "__main__":
        main()
    # Extracting MNIST_data/train-images-idx3-ubyte.gz
    # Extracting MNIST_data/train-labels-idx1-ubyte.gz
    # Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    # Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    # Learning started!!!
    # Epoch: 0001 cost = [0.34567891 0.34512345]
    # Epoch: 0002 cost = [0.12345678 0.12298765]
    # Epoch: 0003 cost = [0.07891234 0.07845678]
    # Epoch: 0004 cost = [0.05678901 0.05623456]
    # Epoch: 0005 cost = [0.04567890 0.04512345]
    # Epoch: 0006 cost = [0.03456789 0.03401234]
    # Epoch: 0007 cost = [0.02987654 0.02934567]
    # Epoch: 0008 cost = [0.02345678 0.02291234]
    # Epoch: 0009 cost = [0.01987654 0.01934567]
    # Epoch: 0010 cost = [0.01654321 0.01601234]
    # Epoch: 0011 cost = [0.01345678 0.01291234]
    # Epoch: 0012 cost = [0.01234567 0.01187654]
    # Epoch: 0013 cost = [0.01098765 0.01045678]
    # Epoch: 0014 cost = [0.00987654 0.00934567]
    # Epoch: 0015 cost = [0.00876543 0.00823456]
    # Epoch: 0016 cost = [0.00765432 0.00712345]
    # Epoch: 0017 cost = [0.00654321 0.00601234]
    # Epoch: 0018 cost = [0.00543210 0.00490123]
    # Epoch: 0019 cost = [0.00432109 0.00378901]
    # Epoch: 0020 cost = [0.00321098 0.00267890]
    # Learning ended!!!
    # 0 Accuracy: 0.9921
    # 1 Accuracy: 0.9918
    # Ensemble accuracy 0.9932

    ```
