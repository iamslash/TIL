# intro

- machine learning이란 다량의 데이터를 이용하여 학습하고 예측하는 것에 대한 학문이다.
  예를 들어서 machine learning을 이용하면 다음과 같은 것을 할 수 있다.
  학생들의 등교시간과 성적에 관한 데이터를 이용하여 새로 전학온 학생의 등교시간을 입력하면
  성적을 예상 할 수 있다.
- machine learning은 회귀분석, 다변량분석, 군집분석, 확률분포 추정,
  마르코프, 은닉마르토크 모델, 서포트 벡터 머신, 베이즈의법칙, 베이즈 확률론,
  베이지언 통계등등 통계학에서 많은 부분들이 인용되었다. 
- machine learning의 종류는 크게 supervised learning, unsupervised learning
  semisupervised learning, reinforcement learning으로 분류할 수 있다.
  다음은 위키피디아가 분류한 것들이다. supervised learning, clustering,
  dimensionality reduction, structured prediction, anomaly detenction,
  neural nets, reinforcement learning
- deep learning은 한개이상의 hidden layer가 존재하는 neural networks를 다루는 학문이다.
  deep의 의미는 hidden layer가 많아지면 점점 깊어진다는 의미이다.
- 다음과 같은 용어들을 중심으로 공부해본다. linear regression with one variable,
  hypothesis function, weight, bias, feature
  loss, gradiant decent algorithm, epoch, cost function,
  MSE (mean squared error), derivative, 
  linear regression with multiple variables, 
  logistic regression, regularization,
  softmax regression (multinomial regression), 
  overfitting, cross entropy, NN (Neural Networks), drop out, activation function,
  sigmoid, ReLU, learning rate, forward propagation, back propagation,
  CNN (Convolutional Neural Networks), RNN (Recurrent Neural Networks)

# learning material

- [모두를 위한 머신러닝/딥러닝 강의](http://hunkim.github.io/ml/)
  - 한글로 제일 쉽게 설명하는 자료이다. 
- [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.amazon.com/gp/product/1491962291/ref=oh_aui_detailpage_o00_s00)
- [밑바닥부터 시작하는 딥러닝](http://www.yes24.com/24/Goods/34970929?Acode=101)
- [machine learning at coursera](https://www.coursera.org/learn/machine-learning)
  - andrew Ng교수의 machine learning강좌
  - 영문이지만 기초를 공부할 수 있다.
- [machine learning note](http://www.holehouse.org/mlclass/)
  - andrew Ng교수의 machine learning강좌 노트
- [deep learning tutorial](http://deeplearning.stanford.edu/tutorial/)
  - standford 대학의 tutorial이다. 코드와 텍스트가 가득하다.
- [c++로 배우는 딥러닝](http://blog.naver.com/atelierjpro/220697890605)
  - 동국대학교 홍정모교수의 한글 딥러닝 강의
- [Andrej Karpathy's Youtube channel](https://www.youtube.com/channel/UCPk8m_r6fkUSYmvgCBwq-sw)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html)
  - 화상인싱을 위한 CNN
- [CS224d: Deep Learning for Natural Language Processing]()
  - 자연어처리를 위한 Deep Learning
- [tensorflow](https://www.tensorflow.org)
- [TensorFlow Tutorials (Simple Examples)](https://github.com/nlintz/TensorFlow-Tutorials)
- [Another TensorFlow Tutorials](https://github.com/pkmital/tensorflow_tutorials)
- [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [Deep learning @Udacity](https://www.udacity.com/course/viewer#!/c-ud730/l-6370362152/m-6379811817)
- [SIRAJ RAVAL'S DEEP LEARNING](https://in.udacity.com/course/deep-learning-nanodegree-foundation--nd101/#)
  - nano degree

# usage

## supervised learning

- supervised learning은 이미 x는 y라는 결론이 도출된 데이터를 이용하여 학습시키는 것이다.
  이미 결론이 도출된 데이터이기 때문에 데이터의 정확성은 높지만 이러한 데이터를 사람이 산출해야
  한다.
- supervised learning의 종류는 classification, regression이 있다. 
- classification은 입력 데이터 x와 출력 데이터 y가 있을때 y가 이산적인 경우
  즉 [0,1,2,..]와 같이 유한한 경우를 말한다.
- classification problem을 해결하기 위한 기법들로 logistic regression,
  KNN (k nearest neighbors), SVM (support vector machines), decision tree
  등이 있다.
- regression은 입력 데이터 x와 출력 데이터 y가 있을때 y가 실수인 경우를 말한다.
  regression problem을 해결하기 위한 기법들로 통계학의 회귀분석 방법중 linear regression
  등이 있다.

## unsupervised learning

- unsupervised learning은 결론이 도출되지 않은 x 데이터를 이용하여 학습시키는 것이다.
  현실 세계의 대부분의 데이터는 결론이 도출되지 않았다.
- unsupervised learning의 종류는 clustering (군집화),
  underlying probability density estimation (분포추정) 등이 있다.

## semisupervised learning

- 다수의 결론이 도출되지 않은 데이터와 약간의 결론이 도출된 데이터를 이용하여 학습시키는 것이다.

## reinforcement learning

- supervised learning과 unsupervised learning는 사람이 학습을 지도 하느냐
  마느냐와 관련이 되어 있지만 reinforcement learning은 조금 다르다.
- 현재의 state (상태)에서 어떤 action (행동)을 취한 것이 최적인지 학습하는 것이다.
  action을 취할때 마다 외부 환경에서 reward (보상)이 주어진다. reward를 최대화
  하는 방향으로 학습이 진행된다.

## linear regression with one variable

- hypothesis function

- cost function

## linear regression with multiple variables

## logistic regression

## softmax regression

## history of deep learning 

- marvin minsky는 1969년 Perceptrons라는 책에서 
  "No one on earth had found a viable way to train"
  이라고 주장했다. XOR을 multi layer perceptron으로 표현은 가능하지만
  학습시키는 불가능하다는 얘기다. 이로써 artificial intelligence분야는
  당분간 사람들의 관심을 떠나게 된다.
- 1974년 1982년 Paul Werbos는 앞서 언급한 marvin minsky의 주장을
  반증 할 수 있는 backpropagation을 발표했지만 사람들의 반응은 냉랭했다.
  심지어는 marvin minsky를 만나서 직접 얘기했지만 marvin minsky의 관심을
  얻지 못했다. 그러나 1986년 Hinton이 발표한 backpropagation은 그렇지 않았다.
  발명보다는 재발견에 해당되지만 전세계 적으로 많은 호응을 얻었다.
- 1995년 LeCun교수는 "Comparison of Learning Algorithms For Handwritten Digit
  Recognition"에서 hidden layer가 많을때 backpropagation과 같이 복잡한
  알고리즘은 문제해결에 효과적이지 못하고 오히려 SVM, RandomForest같은 단순한 
  알로리즘이 효과적이라고 주장한다. neural networks은 다시 침체기로 접어든다.
- 1987년 CIFAR (Canadian Institute for Advanced Research)는 
  deep learning의 침체기 분위기 속에 Hinton교수를 적극 지원했다.
  당시에는 neural networks이라는 키워드가 포함된 논문들은 대부분
  reject되었다.
- 2006년 Hinton교수와 Yoshua Bengio교수는 neural network의 weight를
  잘 초기화 해주면 backpropagation이 가능하다는 논문을 발표한 후 많은
  호응을 얻었다. 그리고 neural networks를 deep learning이라는 새로운
  용어로 재탄생 시켰다.
- imagenet 이라는 대회에서 2010년에 26.2%였던 오류는 2012년 15.3%로 감소하였다.
  이것을 통해 neural networks은 관심을 얻는데 성공한다. 2015년 deep learning을
  이용한 시스템이 오류를 3%까지 감소시킨다.
- Geoffery Hinton교수는 왜 그동안 deep learning이 잘 동작 안했는지 다음과
  같이 4지로 요약했다.
  - Our labeled datasets were thousands of times too small.
  - Our computers were millions of times too slow.
  - We initialized the weights in a stupid way.
    - RBM보다 Xavier방법이 더욱 좋다.
  - We used the wrong type of non-linearity.
    - sigmoid보다 ReLu를 사용하자.
- 이후 알파고를 통해 deep learning은 핫한 기술로 부상한다.

## NN (neural networks)

- marvin minsky의 perceptrons라는 책의 발간 이후 상당 기간동안
  XOR problem은 해결되지 못하고 있었다. 그러나 1986년 Hinton교수를
  통해 해결 방법이 모색되고 neural networks는 다시 관심을 얻게 된다.

- XOR 을 3개의 unit으로 표현해보자. 3개의 unit은 하나의 neural network를 구성한다.

![](xor3units.png)

- chain rule

![](chainrule.png)

- back propagation는 chain rule을 이용하여 구할 수 있다.

- back propagation에 activation function으로 sigmoid를 사용하면
  vanishing gradient가 발생한다. vanishing gradient란
  output layer에서 hidden layer를 거쳐 input layer로 갈수록 입력값의 
  영향을 덜 받게 되는 현상이다. sigmoid보다 ReLU (Rectified Linear Unit)
  을 사용하면 vanishing gradient를 해결 할 수 있다. sigmoid, ReLU를 제외하고도
  tanh, Leaky ReLU, Maxout, ELU등등 Activation Function들이 있다.

- RBM (Restricted Boatman Macine)을 이용하여 weight값을 초기화 하면
  deep learning을 효율적으로 할 수 있다. 그러나 RBM은 너무 복잡하다.
  Xavier initialization 혹은 He's initialization과 같이 간단한
  방법이 더욱 효율적이다.
  
- weight 초기값을 어떻게 설정하느냐는 지금도 활발한 연구 분야이다.

```python
# Xavier initialization
# Glorot et al. 2010
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)

# He et al. 2015
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2)

```

![](backpropagation.png)

## CNN (convolutional networks)

## RNN (recurrent networks)

# reference

- [itex2img](http://www.sciweavers.org/free-online-latex-equation-editor)
 - github markdown에서 수식을 입력할때 사용하자.
  
