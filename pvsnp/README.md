# abstract

* 계산 복잡도는 어떤 문제를 해결하는데 어려움의 정도를 말한다. P, NP, EXP, R 등이 있다.

# Materials

* [16. Complexity: P, NP, NP-completeness, Reductions @ youtube](https://www.youtube.com/watch?v=eHZifpgyH_4)
  * MIT OpenCourseWare 중 [Design and Analysis of Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-videos/) 의 16번 강좌이다.
* [계산복잡도 @ edwith](https://www.edwith.org/introalgorithm/lecture/26441/)
  * MIT 의 OpenCourseWare 중 [Introduction to Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/) 의 강좌이다. 그래프를 이용해서 P, NP, EXP, R 등의 계산복잡도를 설명한다.
* [SNUON_컴퓨터과학이 여는 세계_10.1 P클래스와 NP클래스 문제의 개념_이광근](https://www.youtube.com/watch?v=SW0fRQQYkdA&index=34&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
* [SNUON_컴퓨터과학이 여는 세계_10.2 NP클래스 문제의 예_이광근](https://www.youtube.com/watch?v=6rmJb_6Vx18&index=34&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH#t=5.577024)
* [SNUON_컴퓨터과학이 여는 세계_10.3 NP완전문제의 개념_이광근](https://www.youtube.com/watch?v=J4d2T7XnOT4&index=36&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
* [SNUON_컴퓨터과학이 여는 세계_11.1 건너풀기의 개념과 어려운 문제 판별법_이광근](https://www.youtube.com/watch?v=OBcg0gg1rW8&index=37&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
* [SNUON_컴퓨터과학이 여는 세계_11.2 어려운 문제 현실적으로 풀기: 통밥과 무작위_이광근](https://www.youtube.com/watch?v=Fi8C0Y_FWEQ&index=38&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
* [SNUON_컴퓨터과학이 여는 세계_11.3 어려운 문제의 적당한 해결법_이광근](https://www.youtube.com/watch?v=ZllOMcRSXFA&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH&index=39)
* [Stephen A. Cook: The Complexity of Theorem-Proving Procedures](http://4mhz.de/cook.html)

# P class problems

- P문제는 다항시간에 풀 수 있는 알고리즘이 존재하는 문제를 말한다.
- 정렬문제는 퀵정렬에 의해 O(nlgn)에 해결이 가능하기 때문에 P문제에 해당된다.

# NP class problems

- 대부분의 경우 다항시간이 초과되서 해결되지만 운이 좋으면 (non
  deterministic) 다항시간(polynomial time)에 풀 수 있으며 답이
  주어지는 경우 답을 다항시간에 검증 할 수 있는 문제를 말한다.  예를
  들어 미로찾기를 고민해보자. 운이 좋으면 입구에서 출구까지 다항시간에
  길을 찾을 수 있다. 하지만 대부분의 경우 다항시간에 길을 찾기는
  어렵다. 또한 답이 주어지면 그 답이 맞는지 다항시간에 검증 할 수
  있다. 각 단계마다 어디로 가야할지 답에 기재되어 있기때문에 답을
  검증하는 것은 운이 좋아서 다항시간에 길을 찾아 갈 수 있다는 것과
  같다.
- `NP 문제 ⊃ P 문제` 다 하지만 `NP ≠ P` 라고 할 수는 없다. 아직 `NP ≠ P` 는
  증명이 안되었기 때문이다. 대부분의 사람들은 `NP ≠ P` 일 것 같다고 믿고
  있다. 따라서 P 문제와 NP 문제의 경계는 애매하다.
- 어떤 문제가 주어졌을때 과연 이것이 P 문제인지 NP 문제인지 구별하는
  방법은 중요하다. NP 문제라면 굳이 고생해서 다항시간의 알고리즘을 찾을
  수고를 할 필요가 없기 때문이다.

# NP-complete class problems

- 하나의 문제를 다른 문제로 푸는 것을 reduction(건너풀기) 라고
  한다. complete 은 빠뜨림이 없다는 의미이다.
- Cook-Levin theorem(쿡-레빈 정리) 에 의해 SAT 가 NP-complete 이라는 것은
  증명되었다. 이것은 모든 NP 문제는 다항시간내에 SAT 로 reduction할 수
  있다는 것을 의미한다. 따라서 SAT 문제는 모든 NP 문제 보다 어렵다는
  얘기이다. SAT 문제로 reduction 하기 위해 원래 문제의 입력을 SAT 문제가
  받아들일 수 있게 다항시간에 변형하고 SAT 문제의 결과를 원래 문제에서
  의도한 형태로 다항시간에 변형해야 하기 때문에 시간이 더욱 필요해서
  어렵다고 할 수 있다. SAT 문제는 NP 문제들 중에 가장 어렵다. 마치
  종결자와 같다.
- 만약 SAT 를 다항시간에 해결할 알고리즘이 존재한다면 모든 NP 문제들은
  SAT 로 reduction 해서 다항시간에 해결할 수 있다. SAT 는 마치 기준문제와
  같다. SAT 와 같이 다항시간에 해결할 수 있는 알고리즘이 발견된다면
  모든 NP 문제를 reduction 하여 다항시간에 해결 할 수 있는 그 문제를
  NP-complete 문제 라고
  한다. 
- [Cook-Levin Proof @ youtube](https://www.youtube.com/watch?v=dKS4iDWQVnI&index=1&list=PLS4py2LeEJNDzezHTc0G3EsttsoKWQhGz)
  
# NP-hard class problems

- SAT 처럼 종결자 역할은 하지만 아직 NP문제인지 확인되지 않은 문제를 말한다.

# EXP class problems

- 지수시간에 해결할 수 있는 문제를 말한다.

# R class problems

- 유한한 시간에 해결할 수 있는 문제를 말한다. 
- R 은 Recursion 을 말한다.

# Computation Complexity Diagram

[계산복잡도 @ edwith](https://www.edwith.org/introalgorithm/lecture/26441/) 에 의하면 계산복잡도를 다음과 같이 표현한다.

```
                           R
___________________________|
                   EXP     |
___________________|       |
___________NP      |       |  
___P       |       |       |
___|_______|_______|_______|  
   |       |       |       |
----------------------------------------> Computational Difficulty
          /|       |_______________________
         / |       EXP-Hard
        /  |_______________________________
       /   NP-Hard
      NP-Complete
```

# Question

- 어떤 문제가 주어졌을때 이것이 NP-compelete 인지 어떻게 알 수 있을까?
  - `NP ≠ P` 라고 가정하자. 대부분의 사람들이 그렇게 믿고 있으니깐.  그
    문제에 대해서 대부분의 경우 다항시간을 초과한 알고리즘이
    존재하지만 운이 좋으면 다항시간에 해결 할 수 있는 알고리즘이
    존재하는 것을 발견한다. 즉 그 문제는 NP 문제이다. 이미 증명이 끝난 어떤
    NP-Complete 문제가 그 문제로 reduction 이 가능한지 살펴보자. 
    만약 그렇다면 그 문제는 이미 증명이 끝난 NP-Complete 문제보다 
    어렵다는 것이기 때문에 그 문제는 NP-complete 문제라고 할 수 있다.
    Reduction 은 이와 같이 NP-complete 문제인지 구별하기 위해 사용되기도 한다.