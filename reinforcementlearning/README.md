- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
 
----

# Abstract

reinforcement learning 에 대해 정리한다.

# Materials

* [모두를 위한 딥러닝 @ github](https://hunkim.github.io/ml/)
  * 김훈님의 동영상 강좌이다. DQN 까지 쉽게 설명한다.
* [RLCode와 A3C 쉽고 깊게 이해하기 @ youtube](https://www.youtube.com/watch?v=gINks-YCTBs)
  * RLCode 의 리더 이웅원님의 동영상 강좌이다. 강화학습의 전반적인 내용을 설명한다.
  * [slide](https://www.slideshare.net/WoongwonLee/rlcode-a3c)
* [파이썬과 케라스로 배우는 강화학습 @ wikibooks](http://wikibook.co.kr/reinforcement-learning/)
  * 이웅원님이 저술한 강화학습 책
* [파이썬과 케라스로 배우는 강화학습 @ gitbooks](https://dnddnjs.gitbooks.io/rl/)
  * 이웅원님이 저술한 강화학습 책
* [deeplearning reinforcement learning nanodegree @ udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) 
* [deeplearning reinforcement learning nanodegree - preview @ udacity](https://www.udacity.com/course/ud893-preview) 

# Basic

현재의 상태가 있다. 그리고 상태를 변경할 액션이 있다. 액션은 현재 상태를 다음 상태로 변경하고 보상을 준다. 보상이 많이 주어지는 방향으로 상태를 변경해 가는 것이 강화 학습의 핵심이다. 이것을 dummy q learning 이라고 하고 다음과 같이 수식으로 표현한다.

![](img/dummy_q_learning_eq.png)

```latex
\hat{Q}(s,a) \leftarrow r + \max_{a{}'} \hat{Q}(s{}',a{}') 
```

다음은 openai gym 으로 frozenlake 를 dummy q learning 으로 구현한 것이다.

```python

```

액션을 선택할 때 경우에 따라서 보상이 많은 방향이 아닌 것을 선택해야할 필요가 있다. 이것을 exploit vs exploration 이라고 한다. 상태가 점점 변화될 때마다 보상의 가중치를 낮출 필요가 있다. 이것을 discounted future reward 라고 한다. 이것들을 다음과 같이 수식으로 표현할 수 있다.

![](img/adv_q_learning_eq.png)

```latex
\hat{Q}(s,a) \leftarrow r + \gamma \max_{a{}'} \hat{Q}(s{}',a{}') 
```

다음은 openai gym, frozenlake 를 이용하여 q learning with exploit vs exploration, discounted future reward 을 구현한 것이다.

```python
```

액션의 결과가 미리 정해져 있는 세계를 deterministic world 라고 한다. 액션의 결과가 정해져 있지 않는 세계를 non-deterministic world 라고 한다. non-deterministic world 의 경우 learning rate 를 도입하여 앞으로 보상이 많은 방향만 선택하는 것을 방지해보자. 이것을 다음과 같이 수식으로 표현할 수 있다.

![](img/nondeterministic_q_learning_eq.png)

```latex
\hat{Q}(s,a) \leftarrow (1-\alpha )Q(s,a) + \alpha \left [ r + \gamma \max_{a{}'} Q(s{}',a{}')  \right ]
```

다음은 openai gym, frozenlake 를 이용하여 non-deterministic q learning 을 구현한 것이다.

```python
```

...
