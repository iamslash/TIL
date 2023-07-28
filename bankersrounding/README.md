# Abstract

banker's rounding 에 대해 정리한다. 

# Materials

* [은행가의 반올림 (Banker’s rounding)](https://www.freeism.co.kr/wp/archives/1792)
* [Bankers Rounding](https://wiki.c2.com/?BankersRounding)

# Basic

`0.5` 를 반올림하면 `0` 이다. `1.5` 을 반올림하면 `2` 이다. 왜??? 

```py
>>> round(0.5)
0
>>> round(1.5)
2
>>> round(2.5)
2
>>> round(3.5)
4
>>> round(4.5)
4
```

이것을 banker's round 라고 한다. 규칙은 가장 가까운 짝수로 반올림한다.

`12.0` 에서 `13.0` 을 9 개의 숫자로 나누자. 즉, "12.1, 12.2, ..., 12.9" 와 같이 9 개의 숫자를 생각해 보자. standard rounding 의 경우 4 개의 숫자는 12 가 되고 5 개의 숫자는 13 이 된다. 1 개의 숫자가 공평하지 못하다.

`12.0` 에서 `14.0` 을 18 개의 숫자로 나누자. 즉, "12.1, 12.2, ..., 13.9" 와 같이 18 개의 숫자를 생각해 보자. banker's rounding 의 경우 9 개의 숫자는 12 가 되고 9 개의 숫자는 14 가 된다. 공평하다.
