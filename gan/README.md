- [Abstract](#abstract)
- [Materials](#materials)
- [Implementation for Intuition](#implementation-for-intuition)
- [Expectation Function](#expectation-function)
- [Entropy](#entropy)
- [Cross Entropy](#cross-entropy)
- [KLD (Kullback–Leibler divergence)](#kld-kullback%E2%80%93leibler-divergence)
- [JSD (Jensson Shannon Divergence)](#jsd-jensson-shannon-divergence)
- [Posterior](#posterior)
- [Likelihood](#likelihood)
- [Prior](#prior)
- [Bayes' Rule](#bayes-rule)
- [Objective Function](#objective-function)
- [Poor Gradient in Early Training](#poor-gradient-in-early-training)
- [Global Optimality of `P_{g} = P_{data}`](#global-optimality-of-pg--pdata)
- [Convergence of Algorithm 1](#convergence-of-algorithm-1)
- [Simple GAN by keras](#simple-gan-by-keras)
- [DCGAN by keras](#dcgan-by-keras)

-----

# Abstract

- discriminator는 이미지데이터를 입력받아 진짜인지 가짜인지 출력하기
  위해 학습시킨다.
- generator는 임의의 데이터를 입력받아 이미지데이터를 출력한다. 
- GAN 은 generator 가 생성한 이미지를 discriminator 에 주어 
  discriminator 의 출력이  1 에 가까워지도록 학습한다.
- GAN 이 제대로 학습되었다면 generator 는 진짜에 가까운 이미지데이터를 
  생성할 것이다.
- 다양한 GAN 을 공부해 보자. DCGAN (Deep Convolutional GAN), LSGAN (Least Squares GAN), SGAN
  (Semi-Supervised GAN), ACGAN (Auxiliary Classifier GAN), CycleGAN, StackGAN

# Materials

- [한국어 기계 학습 강좌 @ kaist](https://aailab.kaist.ac.kr/xe2/page_GBex27/)
  - 문일철 교수님의 머신러닝강좌. MLE, MAP, 확률분포등 유용한 내용들이 있음
- [Generative Adversarial Nets @ arxiv](https://arxiv.org/pdf/1406.2661.pdf)
  - Ian J. Goodfellow 논문
- [GAN Tutorial @ youtube](https://www.youtube.com/watch?v=uQT464Ms6y8&index=1&list=RDuQT464Ms6y8)
  - [ppt](https://drive.google.com/file/d/0B377f9tIGAcwdVd1Z3dCX1lBTlE/view)
  - 최윤제님이 pytorch를 이용하여 code를 먼저 직관적으로 만들어보고 [Generative Adversarial Nets @ arxiv](https://arxiv.org/pdf/1406.2661.pdf) 논문을 자세히 설명함
- [1시간만에 GAN(Generative Adversarial Network) 완전 정복하기 @ naver](http://tv.naver.com/v/1947034)
  - [src](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py)
  - [ppt](https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network)
  - 최윤제님의 다른 설명
- [PR-001: Generative adversarial nets by Jaejun Yoo (2017/4/13) @ youtube](https://www.youtube.com/watch?v=L3hz57whyNw)
  - 유재준님의 [Generative Adversarial Nets @ arxiv](https://arxiv.org/pdf/1406.2661.pdf) 설명
- [GAN by Example using Keras on Tensorflow Backend](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
  - [src](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py)
  - [쌩(?!)초보자의 Python 케라스(Keras) GAN 코드 분석 (draft)](http://leestation.tistory.com/776)
- [Chara Tsukuru GAN: RPG character generator @ github](https://github.com/almchung/chara-tsukuru-gan)
  - RPG 2D 캐릭터 스프라이트를 생성하는 DCGAN
- [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation @ arxiv](https://arxiv.org/abs/1711.09020)
  - 최윤제님 논문
  - [starGAN @ youtube](https://www.youtube.com/watch?v=D80h0MfaspA)
  - [한글설명](http://www.modulabs.co.kr/?module=file&act=procFileDownload&file_srl=20159&sid=16dcd07bb230645a7a9b9271ee6a04ac&module_srl=17958)
  
# Implementation for Intuition

* Discriminator Block

`Discriminator` 는 진짜 이미지의 데이터 `x` 를 입력받아 그것이
진짜 이미지일 확률 `D(x)` 를 출력한다. `D(x)` 는
확률이기 때문에 `[0, 1]` 의 값이다.

```
x -> D -> D(x)
```

* Adversarial Diagram

`Generator` 는 랜덤값 `z` 를 입력받아 가짜이미지 `G(z)` 를
출력하고 이것을 다시 `Discriminator` 의 입력으로 넘겨준다.
`Discriminator` 는 가짜 이미지 데이터 `G(z)` 를 입력받아
이것이 진짜 이미지일 확률 `D(G(z))` 를 출력한다. `D(G(z))` 는
확률이기 때문에 `[0, 1]` 의 값이다.


```
z -> G -> G(z) -> D -> D(G(z))
```

pytorch 로 구현된 예를 보고 직관적으로 접근해 보자.

[main.py](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py)

```py
...
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
...        
```

# Expectation Function

기대값 `E(X)` 을 어떻게 구하는지 설명한다. 완벽한 세계에서 주사위를 던지는 상황을 생각해 보자. 확률변수 X 와 P(X) 는 다음과 같다.

| X | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| P(X) | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

다음은 확률변수 X 가 Discrete Space 에 속하는 경우와 X 가 continuous Space 에 속할 때의 `E(x)` 이다.

![](exp_x_discrete.png)

```latex
\begin{align*}
E_{x \sim  p(x)} [X] &= \sum _x p(x) X \\
                     &= \int _x p(x) X dx
\end{align*}
```

확률변수 `X` 가 함수 일수도 있다. 확률변수가 `f(x)` 라고 하면 `E(f(x))` 는 다음과 같다.

![](exp_func_discrete.png)

```latex
\begin{align*}
E_{x \sim  p(x)} [f(x)] &= \sum _x p(x) f(x) \\
                        &= \int _x p(x) f(x) dx
\end{align*}
```

# Entropy

* [참고](http://t-robotics.blogspot.com/2017/08/26-entropy.html)
* [Entropy & Information Theory](https://hyeongminlee.github.io/post/prob001_information_theory/)

정보를 최적으로 인코딩하기 위해 필요한 bit 의 수를 말한다.

예를 들어서 오늘이 무슨 요일인지 bit 로 표현해보자. 요일의 개수가
모두 7개이니까 3비트가 필요하다. (log_{2}7 = 2.8073)

```
월 : 000
화 : 001
수 : 010
...
```

만약 표현할 정보가 나타날 확률이 다르다고 해보자. 예를 들어
40 개의 문자 (A, B, C, D, ..., Z, 1, 2, 3, ..., 14) 를
bit 로 표현해보자. 40 개이기 때문에 6 bit 가 필요하다. (log_{2}40 = 5.3219)

그런데 A, B, C, D 가 발생할 확률이 각각 22.5% 라고 해보자.
모두 합하면 90% 확률이다. 6개의 비트를 모두 사용할 필요가 없다.

첫번째 비트를 A, B, C, D 인지 아닌지를 표현하도록 하자. 만약 첫번째 비트가
1 이면 추가로 2 비트만 있으면 A, B, C, D 를 구분할 수 있다. 만약 첫번째 비트가
0 이면 추가로 6 bit 가 필요하다. (log_{2}36 = 6) 결과적으로 필요한 비트는 3.3 비트가 된다. 확률을 고려했을 때 평균적으로 필요한 비트의 수를 엔트로피라고 한다.

```
0.9 * 3 + 0.1 * 6 = 3.3 bit
```

Entropy 의 공식을 다음과 같이 정리할 수 있다.

![](entropy_eq.png)

```latex
\begin{align*}
H &= \sum _{i} p_{i} I(s_{i}) \\
  &= \sum _{i} p_{i} \log _{2} (\frac {1}{p_{i}}) \\
  &= - \sum _{i} p_{i} \log _{2} (p_{i})
\end{align*}
```

`I(s_{i})` 를 information gain 이라고 할 수 있다. information gain 은 획득한 정보의 양을 얘기하는데 이것은 그 정보가 나타날 확률에 반비례한다. 예를 들어 김씨가 임씨보다 많다.
어떤 사람이 임씨일 경우 information gain 은 김씨일 경우보다 높다. 더우 희귀한 성이기 
때문에 그 사람에 대해 더욱 많은 정보를 획득했다고 할 수 있다.

# Cross Entropy

* [참고](http://t-robotics.blogspot.com/2017/08/27-cross-entropy-kl-divergence.html)

Entropy 공식을 다시 살펴보자. 

![](entropy_eq.png)

`p` 를 실제확률 `q` 를 예측확률 이라고 하자. 다음과 같이
cross entropy 를 정의할 수 있다.

![](cross_entropy_eq.png)

```latex
```

# KLD (Kullback–Leibler divergence)

* [참고](http://t-robotics.blogspot.com/2017/08/27-cross-entropy-kl-divergence.html)
* [Kullback-Leibler Divergence & Jensen-Shannon Divergence](https://hyeongminlee.github.io/post/prob002_kld_jsd/)

우리가 데이터의 분포를 추정했을 때 얼마나 잘 추정한 것인지 측정하는
방법이 필요하다. KLD 는 서로 다른 확률분포의 차이를 측정하는 척도이다. 
KLD 가 작다면 좋은 추정이라 할 수 있다.

먼저 아이템 `s_{i}` 가 갖는 information gain 은 다음과 같다.

![](kld_information_gain.png)

```latex
I_{i} = - \log (p_{i})
```

원본 확률 분포 p 와 근사된 분포 q 에 대하여 i 번째 아이템이
가진 정보량 (information gain) 의 차이 (정보손실량) 은 다음과 같다.

![](kld_information_gain_delta.png)

```latex
\Delta I_{i} = - \log (p_{i}) - \log (q_{i})
```

p 에 대하여 이러한 정보 손실량의 기대값을 구한 것이 바로 KLD 이다.

![](kld_eq.png)

```latex
\begin{align*}
D_{KL}(p||q) &= E[\log(p_{i}) - \log(q_{i})] \\
             &= \sum _{i} p_{i} \log \frac {p_{i}}{q_{i}}
\end{align*}
```

그러나 KLD 는 symmetric 하지 않다. 즉 `D_{KL}(P||q) != D_{KL}(q||p)` 이다.

# JSD (Jensson Shannon Divergence)

* [참고](https://hyeongminlee.github.io/post/prob002_kld_jsd/)

KLD 는 symmetric 하지 않다. 즉 `D_{KL}(P||q) != D_{KL}(q||p)` 이다.
KLD 를 symmetric 하게 개량한 것이 JSD 이다.

![](jsd_eq.png)

```latex
JSD(p, q) = \frac {1}{2} D_{KL} (p || \frac {p + q}{2}) + D_{KL} (q || \frac {p + q}{2})
```

# Posterior

* [참고](https://hyeongminlee.github.io/post/bnn001_bayes_rule/)

물고기가 주어졌을 때 이것이 농어인지 연어인지 구분하는 문제를 살펴보자.
피부색의 밝기를 `x` 라고 물고기의 종류를 `w` 라고 하자. 물고기가 농어일
사건을 `w = w_{1}` 연어일 사건을 `w = w_{2}` 라고 하자.

그렇다면 물고기의 피부 밝기가 0.5 일 때 그 물고기가 농어일 확률은 다음과 
같이 표현할 수 있다.

![](posterior_ex.png)

```latex
\begin{align*}
P(w = w_{1} | x = 0.5) = P(w_{1} | x = 0.5)
\end{align*}
```
이제 임의의 `x` 에 대해 `P(w_{1}|x)` 와 `P(w_{2}|x)` 의 값이 주어지면
다음과 같은 방법을 통해 농어와 연어를 구분할 수 있다.

* `P(w_{1}|x) > P(w_{2}|x)` 라면 농어로 분류하자.
* `P(w_{2}|x) > P(w_{1}|x)` 라면 연어로 분류하자.

`P(w_{i}|x)` 를 사후확률 (Posterior) 라고 한다.

# Likelihood

* [참고](https://hyeongminlee.github.io/post/bnn001_bayes_rule/)

물고기를 적당히 잡아서 데이터를 수집해 보자. `P(x|w_{1}` 에 해당하는 농어의
피부밝기 분포와 `P(x|x_{2}` 에 해당하는 연어의 피부밝기 분포를 그려보자.
이렇게 관찰을 통해 얻은 확률 분포 `P(x|w_{i})` 를 가능도 (likelihodd)
라고 부른다.

# Prior

* [참고](https://hyeongminlee.github.io/post/bnn001_bayes_rule/)

`x` 와 관계없이 애초에 농어가 잡힐 확률 `P(w_{1})`, 연어가 잡힐 확률 `P(w_{2})`
를 사전확률 (Prior) 라고 한다. 이미 갖고 있는 사전 지식에 해당한다.

# Bayes' Rule

* [참고](https://hyeongminlee.github.io/post/bnn001_bayes_rule/)

우리의 목적은 Posterior `P(w_{i}|x)` 를 구하는 것이다. 이 것은 Likelihood `(P(x|w_{i})` 와 Prior `P(w_{i})` 를 이용하면 구할 수 있다.

![](bayes_rule.png)

```latex
\begin{align*}
P(A, B) &= P(A|B) B(B) = P(B|A) P(A) \\
P(A| B) &= \frac {P(B|A)P(A)}{P(B)} = \frac {P(B|A)P(A)}{\sum _{A} P(B|A)P(A)} \\
P(w_{i} | x) &= \frac {P(x | w_{i})P(w_{i})}{\sum _{j} P(x|w_{j})P(w_{j})}
\end{align*}
```

좌변은 Posterior 이고 우변의 분자는 Likelihood 와 Prior 의 곱이다. 분모는 Evidence 라고 부른다. 이것 또한 Likelihood 와 Prior 들을 통해 구할 수 있다. 이러한 식을
Bayes' Rule 또는 Bayesian Equation 등으로 부른다.

# Objective Function

다음과 같은 `objective function` 에 대해 `V(D, G)` 를 최소로 만드는 `G` 와 최대로 만드는 `D` 를 찾으면 실제 이미지를 생성하는 `GAN` 을 완성할 수 있다.

![](gan_objective_function.png)

```latex
\begin{align*}
\min _{G} \max _{D} V(D, G) = \mathbb{E}_{x \sim p _{data} (x)}[\log D(x)] + \mathbb{E}_{z \sim p _{z} (z)}[\log (1 - D(G(z)))]
\end{align*}
```

`D` 의 관점에서 생각해 보자. `D(x) = 1` 이고 `D(G(z)) = 0` 일 때 `V(G, D)` 는 최대이다. `D(x) = 1` 일 때는 진짜를 진짜로 판별했을 때를 의미하고 `D(G(z)) = 0` 일 때는 가짜를 가짜로 판별했을 때를 의미한다.

`G` 의 관점에서 생각해 보자. `V(D, G)` 의 첫번째 항은 `G` 와 관련없기 때문에 무시해도 된다. 두번째 항을 살펴보자. `D(G(z)) = 1` 일 때 `V(D, G)` 는 최소이다.

즉 다음과 같은 수식을 만족하는 `G` 를 구해야 한다.

![](gan_objective_eq_G.png)


```latex
\min _{G} \mathbb{E}_{z \sim p _{z} (z)}[\log (1 - D(G(z)))]
```

`x = D(G(z))` 라고 생각하면 위의 식을 만족하는 `G` 를 찾는 것은 `y = log(1-x)` 그래프에서 최소의 기울기를 갖는 `x` 를  찾는 것과 같다. 이때 `x` 의 범위는 `[0, 1]` 임을 유의하자. `x` 가 0일 때 이미 기울기가 최소이기 때문에 학습이 잘 일어나지 않는다. heuristic 하게 생각해서 최대 기울기 부터 시작하는 그래프를 생각해보자.

![](gan_graph_log_1-x.png)

그래서 다음수식을 생각해 냈다. 다음의 식을 최대로 하는 `G` 를 찾는 것은 앞서 언급한 수식을 최소로 하는 `G` 를 찾는 것과 같다.

![](gan_objective_eq_G_max.png)

```latex
\max _{G} \mathbb{E}_{z \sim p _{z} (z)}[\log D(G(z))]
```

 실제로 `y = log(x)` 그래프에서 `x` 가 0일 때 최대의 기울기를 갖는다. 학습 초반에 Generator 가 빨리 벋어 나려고 한다???

![](gan_graph_log_x.png)

그럼 이것을 어떻게 구현할 것인가? `sigmoid cross entropy loss` 를 이용하여 다음과 같은 수식을 만들자.

![](gan_sigmoid_cross_entropy_loss.png)

```latex
```

`y = 1` 을 하면 다음과 같다.


![](gan_sigmoid_cross_entropy_loss_y_1.png)

```latex
```

결국 `logD(G(z))` 의 기대값을 최대로 하는 `G` 를 찾는 것은 `-logD(G(z))` 의 기대값을 최소로 하는 `G` 를 찾는 것과 같다. 이것을 `Poor Gradient in Early Training` 이라고 한다.

# Poor Gradient in Early Training

# Global Optimality of `P_{g} = P_{data}`

`G` 를 고정하고 최적화된 `D` 를 얻어보자.

# Convergence of Algorithm 1

`V(D, G)` 를 최대로 하는 `D` 를 찾았다고 가정하면 objective function 은 다음과 같은 식과 동치이다.

![](gan_jsd_eq.png)

```latex
```

# Simple GAN by keras

```py
```

# DCGAN by keras 

```py
```
