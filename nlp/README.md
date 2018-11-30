# Abstract

자연어 처리에 대해 정리한다.

# Materials

* [[토크ON세미나] 딥러닝을 활용한 자연어 처리 기술 실습 1강 | T아카데미 @ youtube](https://www.youtube.com/watch?v=F8b0jGyZ_W8&list=PL9mhQYIlKEhdorgRaoASuZfgIQTakIEE9)
  * [src](https://github.com/hugman/deep_learning)
* [Natural Language Processing Tasks and Selected References @ github](https://github.com/Kyubyong/nlp_tasks)
  * 박규병님의 nlp links
* [kakao khaiii @ github](https://github.com/kakao/khaiii)
  * kakao 에서 개발한 형태소분석기
* [Modern Deep Learning Techniques Applied to Natural Language Processing](https://nlpoverview.com/)
  * nlp deeplearning
  * [src](https://github.com/omarsar/nlp_overview)
* [NLTK 로 배우는 자연어처리](https://blog.naver.com/bcj1210/221144598072)
* [NLTK with Python 3 for Natural Language Processing @ youtube](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
* [Deep Natural Language Processing course offered in Hilary Term 2017 at the University of Oxford](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/README.md)
* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
  * [2017 video](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)


# Sentiment Analysis

문장이 담고 있는 감성을 분석한다. 예를 들어 `유일한 단점은 방안에 있어도 들리는 소음인데요.` 와 같은 문장이 입력으로 들어오면 출력으로 `POS, OBJ, NEG, NEU` 내보낸다. `n21` 문제이다.

# Named Entity Recognition

문장이 담고 있는 단어들의 개체명을 분석한다. 예를 들어 개체이름을 다음과 같이 5개로 설계해보자.

```
OG : Organization
DT : Date
TI : Time
LC : Location
PS : Person
```

앞서 언급한 클래스들을 `Tag` 라고 하였을 때 `Tag` 의 시작음절을 `B-Tag`, `Tag` 의 중간음절을 `I-Tag`, `Tag` 와 상관없는 음절을 `O` 로 표기하기로 하자.

`지난달 27일부터 매일 오후 4시에` 라는 입력이 들어왔다고 해보자. 다음과 같이 NRE 할 수 있다.

| 지 | 난 | 달 |   | 2 | 7 | 일 | 부 | 터 |   | 매 | 일 |   | 오 | 후 |   | 4 | 시 | 에 |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| `B-DT` | `I-DT` | `I-DT` | `I-DT` | `I-DT` | `I-DT` | `I-DT` | `O` | `O` | `o` | `B-DT` | `I-DT` | `O` | `B-TI` | `I-TI` | `I-TI` | `I-TI` | `I-TI` | `O` | 