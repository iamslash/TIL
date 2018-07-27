# Abstract


- discriminator는 이미지데이터를 입력받아 진짜인지 가짜인지 출력하기
  위해 학습시킨다.
- generator는 임의의 데이터를 입력받아 이미지데이터를 출력한다. 출력된
  이미지데이터는 다시 discriminator의 입력데이터로 사용되어
  generator가 생성한 이미지가 진짜가 될 수 있도록 ganerator를
  학습시킨다.
- [1시간만에 GAN(Generative Adversarial Network) 완전 정복하기](http://tv.naver.com/v/1947034)
  - [src](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan)
  - [ppt](https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network)
  - pytorch를 이용한 설명
- DCGAN (Deep Convolutional GAN), LSGAN (Least Squares GAN), SGAN
  (Semi-Supervised GAN), ACGAN (Auxiliary Classifier GAN), NLP
  (natural language processing)
- CycleGAN, StackGAN

# Materials

* [쌩(?!)초보자의 Python 케라스(Keras) GAN 코드 분석 (draft)](http://leestation.tistory.com/776)