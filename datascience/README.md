# Abstract

![data science venn diagram](https://static1.squarespace.com/static/5150aec6e4b0e340ec52710a/t/51525c33e4b0b3e0d10f77ab/1364352052403/Data_Science_VD.png?format=1500w)

- 데이터 과학을 이해하려면 다음과 같은 주제들을 공부해야한다.
  - 데이터시각화, 선형대수, 통계, 확률, 가설과 추론, 경사 하강법, 기계학습
  - k-근접이웃, 나이브 베이즈, 단순 선형회귀, 다중 선형회귀, 로지스틱 회귀
  - 의사결정나무, 신경망, 군집화, 자연어처리, 추천시스템
- learning material 위주로 공부 해본다.

# Learning Materials

- [데이터 사이언스 스쿨 (파이썬 버전)](https://datascienceschool.net/view-notebook/661128713b654edc928ecb455a826b1d/)
  - 오프라인강좌의 커리큘럼의 일부이다. numpy, pandas, scikit-learn등 한글 설명이 좋다.
- [따라하며 배우는 데이터 과학 : 실리콘벨리 데이터 과학자가 알려주는](http://www.yes24.com/24/Goods/44184320?Acode=101)
  - 완전 입문서이다. 통계학을 공부해야 하는 동기부여가 된다.
- [기초통계학 at youtube](https://www.youtube.com/playlist?list=PLsri7w6p16vs-vfUgweXPjEEhwXjjPSHq)
  - 노경섭의 제대로 시작하는 기초통계학 동영상 강좌
- [논문통계분석 at youtube](https://www.youtube.com/watch?v=8PT4AKrKjFo&list=PLsri7w6p16vuIphjhykx6UwOb6ICK0HVi)
  - 노경섭의 제대로 알고쓰는 논문 통계분석 동영상 강좌
- [openintro](https://www.openintro.org/stat/videos.php)
  - 통계학을 쉽게 알려주는 비디오 강좌
- [Thinks Stats](http://greenteapress.com/thinkstats/)
  - [code](https://github.com/AllenDowney/ThinkStats2)
  - [한글](http://fliphtml5.com/dvlr/gyzu/basic)
  - python 프로그래머를 위한 통계학. 공짜.
- [21세기 통계학을 배우는 방법](http://statkclee.github.io/window-of-statistics/)
  - 매우 설득력 있는 통계 로드맵.
- [빅데이터의 통찰력을 키워주는 엑셀 Quick Start](https://www.inflearn.com/course/%EC%97%91%EC%85%80-%EA%B0%95%EC%A2%8C/)
  - 엑셀을 이용해서 기초통계분석과 시각화를 할 수 있다.
- [헬로 데이터 과학](http://www.kangcom.com/sub/view.asp?sku=201602122364)
  - [데이터과학자료모음](http://www.hellodatascience.com/?page_id=7)
- [빅데이터강좌 in dbguide](http://cyber.dbguide.net/lecture.php?code=AA017)
  - 한글 무료 강좌가 가득하다. 좀 오래된 내용이 단점이다. 재미가 없다. 하지만 방향을 위해 본다.
- [밑바닥부터 시작하는 데이터 과학](http://www.kangcom.com/sub/view.asp?sku=201605307751)
  - 넓고 얕게 다룬다. 공부의 방향을 정할 수 있다.
  - [code](https://github.com/Insight-book/data-science-from-scratch?files=1)
- [앞으로 데이터 분석을 시작하려는 사람을 위한 책](http://www.aladin.co.kr/shop/wproduct.aspx?ItemId=40672590&ttbkey=ttbcloud092006002&COPYPaper=1)
  - 일본인이 쓴 책이다. 공부의 방향을 정할 수 있다.
- [intro to data science](https://classroom.udacity.com/courses/ud359)
  - beginner에게 어울리는 강좌이다.
- [Data Science at Scale Specialization](https://www.coursera.org/specializations/data-science)
  - 워싱턴대학교 데이터과학 입문
- [Statistics One](https://www.youtube.com/watch?v=VJlpQs4a5LI&list=PLgIPpm6tJZoTlY4A-xikgjXmlscqduP5k)
  - 프린스턴대 통계 1
- [D3 tutorial](http://alignedleft.com/tutorials)
  - 데이터시각화를 위한 D3 튜토리얼

# datasets

- [공공데이터포털](https://www.data.go.kr/)
  - 전국 신규 민간 아파트 분양가격 동향
- [kaggle](https://www.kaggle.com/)
  - 유용한 데이터들과 함께 문제들이 제공된다.

# numpy
# pandas
# scikit-learn
# probability

- joint probability 결합확률
  - 사건 A와 B가 동시에 발생할 확률 
  - `P(A∩B) or P(A,B)`
  
- conditional probability 조건부확률
  - 사건 B가 사실일 경우 사건 A에 대한 확률 
  - `P(A|B) = P(A,B) / P(B)`

- independent event 독립사건
  - 사건 A와 사건 B의 교집합이 공집합인 경우 A와 B는 서로 독립이라고 한다. 
  - `P(A,B) = P(A)P(B)`
  - `P(A|B) = P(A,B) / P(B) = P(A)P(B) / P(B) = P(A)`
  
- Bayes' theorem 베이즈 정리
  - `P(A|B) = P(B|A)P(A) / P(B)`
  - 증명
```
P(A,B) = P(A|B)P(B)
P(A,B) = P(B|A)P(A)
P(A|B)P(B) = P(B|A)P(A)
P(A|B) = P(B|A)P(A) / P(B)
```

- random variable 확률변수
  - 확률이라는 규칙을 가지면서 변하는 수치를 표현한다. 
  - 보통 대문자 로 표현한다.
  - 확률변수가 가질 수 있는 값의 형태에 따라 discrete random
    variable(이산형 확률변수), continuous variable(연속 확률변수)로
    구분한다.

```
예) 동전던지기에서 앞면이 나타나는 경우의 수로 정의되는 확률변수 X
    X = 동전전지기에서 나타나는 앞면의 수
    X = 0, 1 (동전던지기에서 앞면은 1번 또는 0번 나타남을 표현하는 것임)
```

- CDF(cumulative distribution function) 누적분포함수
  - `F(x) = P({X < x}) = P(X < x)`

- PDF(probability density function) 확률밀도함수
  - CDF는 분포의 형상을 직관적으로 이해하기 힘들다. 어떤 확률 변수
    값이 더 자주 나오는지에 대한 정보를 알기 위해 상대적인 확률 분포
    형태만을 보기위한 것

- PMF(probability mass function) 확률질량함수
