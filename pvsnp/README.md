# abstract

- P, NP 문제에 대해 적어보자.

# P class problems

- P문제는 다항시간에 풀 수 있는 문제를 말한다.
- 정렬문제는 퀵정렬에 의해 O(nlgn)에 해결이 가능하기 때문에 P문제에 해당된다.

# NP class problems

- 대부분의 경우 다항시간이 초과되서 해결되지만 운이 좋으면 다항시간에 풀 수
  있으며 답이 주어지는 경우 답을 다항시간에 검증 할 수 있는 문제를 말한다.
  예를 들어 미로찾기를 고민해보자. 운이 좋으면 입구에서 출구까지 다항시간에
  길을 찾을 수 있다. 하지만 대부분ㄷ 다항시간에 길을 찾기는 어렵다. 또한
  답이 주어지면 그 답이 맞는지 다항시간에 검증 할 수 있다. 다시
  미로찾기의 경우 답이 주어졌을때 단계별로 따라 가다 보면 출구로
  가는지 가지 않는지 다항시간에 확인 할 수 있다. 답이 주어져서
  단계별로 답으로 간다는 경우는 운지 좋다는 것과 같은 의미이다.
- 어떤 문제가 주어졌을때 과연 이것이 P문제인지 NP문제인지 구별하는
  방법은 중요하다. NP문제라면 굳이 고생해서 다항시간의 알고리즘을 찾을
  수고를 덜 수 있기 때문이다.
- NP ≠ P 는 아직 증명되지 않은 유명한 문제이다. 대부분의 사람들은
  그럴 것 같다고 믿고 있다. 따라서 P문제와 NP문제의 범주 경계는
  애매하다. NP문제 ⊃ P문제는 맞지만 NP ≠ P 라고 할 수는 없다. 아직
  NP ≠ P는 증명이 안되었기 때문이다.

# NP-complete class problems

- 하나의 문제를 다른 문제로 푸는 것을 reduction(건너풀기)라고
  한다. complete은 빠뜨림이 없다는 의미이다. NP문제들중 SAT는 가장
  어려운 문제 인것이 발견된다. 그외 많은 NP문제들은 SAT로 reduction할
  수 있다. 만약 SAT를 다항시간에 해결할 알고리즘이 있다면 그외 많은
  NP문제들은 SAT로 다항시간에 전환해서 건너풀기하면 다항시간에 해결할
  수 있다. 이때 종결자에 해당하는 SAT를 다항시간에 해결할 수 있다면
  빠짐 없이 건너풀기를 통해 다항시간에 해결 할 수 문제들을
  NP-complete문제라고 한다.  이것은 2-3페이의 논문으로 이미
  증명되었다.
  
# NP-hard class problems

- 종결자 역할은 하지만 NP문제인지 확인되지 않은 문제를 말한다.

# 어떤 문제가 주어졌을때 P문제 바깥의 문제인지 구별하는 방법

- NP ≠ P 라면 ???

# references

- [SNUON_컴퓨터과학이 여는 세계_10.1 P클래스와 NP클래스 문제의 개념_이광근](https://www.youtube.com/watch?v=SW0fRQQYkdA&index=34&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_10.2 NP클래스 문제의 예_이광근](https://www.youtube.com/watch?v=6rmJb_6Vx18&index=34&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH#t=5.577024)
- [SNUON_컴퓨터과학이 여는 세계_10.3 NP완전문제의 개념_이광근](https://www.youtube.com/watch?v=J4d2T7XnOT4&index=36&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_11.1 건너풀기의 개념과 어려운 문제 판별법_이광근](https://www.youtube.com/watch?v=OBcg0gg1rW8&index=37&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_11.2 어려운 문제 현실적으로 풀기: 통밥과 무작위_이광근](https://www.youtube.com/watch?v=Fi8C0Y_FWEQ&index=38&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_11.3 어려운 문제의 적당한 해결법_이광근](https://www.youtube.com/watch?v=ZllOMcRSXFA&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH&index=39)
