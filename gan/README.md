# Abstract

- discriminator는 이미지데이터를 입력받아 진짜인지 가짜인지 출력하기
  위해 학습시킨다.
- generator는 임의의 데이터를 입력받아 이미지데이터를 출력한다. 출력된
  이미지데이터는 다시 discriminator의 입력데이터로 사용되어
  generator가 생성한 이미지가 진짜가 될 수 있도록 ganerator를
  학습시킨다.
- 다양한 GAN 을 공부해 보자. DCGAN (Deep Convolutional GAN), LSGAN (Least Squares GAN), SGAN
  (Semi-Supervised GAN), ACGAN (Auxiliary Classifier GAN), CycleGAN, StackGAN

# Materials

- [GAN Tutorial @ youtube](https://www.youtube.com/watch?v=uQT464Ms6y8&index=1&list=RDuQT464Ms6y8)
  - [ppt](https://drive.google.com/file/d/0B377f9tIGAcwdVd1Z3dCX1lBTlE/view)
  - pytorch를 이용하여 code를 먼저 직관적으로 만들어보고 논문을 자세히 설명함
- [1시간만에 GAN(Generative Adversarial Network) 완전 정복하기 @ naver](http://tv.naver.com/v/1947034)
  - [src](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan)
  - [ppt](https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network)
  - pytorch를 이용한 설명
- [쌩(?!)초보자의 Python 케라스(Keras) GAN 코드 분석 (draft)](http://leestation.tistory.com/776)
  
# Implementation for Intuition

keras 로 간단히 구현해보고 GAN 을 직관적으로 접근해보자.

```
```

# Objective Function

```
```

# Expectation Function

```
```

# JSD (Jensson Shannon Divergence)

```
```

# DCGAN (Deep Convolution Generative Adversarial Network)

```py
```
