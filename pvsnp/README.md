# abstract

# P class problems

- 다항시간에 풀 수 있는 문제
- quick sort, bubble sort

# NP class problems

- 운이 좋으면 다항시간에 풀 수 있는 문제이다. 예를 들어 미로찾기를 고민해보자. 운이 좋으면
  입구에서 출구까지 다항시간에 길을 찾을 수 있다.
- 답이 주어지면 그 답이 맞는지 다항시간에 검증 할 수 있다. 예를 들어 미로찾기의 답이 주어졌을때 
  단계별로 따라 가다 보면 출구로 가는지 가지 않는지 다항시간에 확인 할 수 있다.
- 현실의 많은 문제들은 NP 문제들이다.
- hamiltonian path, TSP, SAT
- NP문제중에 SAT문제는 가장 어렵다. SAT를 다항시간안에 해결 할 수 있다면 모든 NP문제는
  다항시간안에 풀 수 있다.
- NP와 P의 경계는 모호하다. 뚜렷하지 않다. NP ≠ P 가 아직 증명되지 않았기 때문이다.
  그러나 대부분의 사람들은 NP ≠ P 라고 생각한다.
- NP class problems ⊃ P class problems. 하지만 NP ≠ P 가 사실인지 아닌지는
  아직 증명이 안되어 있다. 대부분의 사람들이 그렇게 믿고 싶을 뿐이다.

# NP-complete class problems

- 하나의 문제를 다른 문제로 푸는 것을 problem reduction(건너풀기)라고 한다.
- complete은 빠뜨림이 없다는 의미이다. SAT를 다항시간에 해결할
  알고리즘이 있다면 NP-complete의 모든 문제들은 빠뜨림 없이 모두
  다항시간에 해결 할 수 있다. 이것은 2-3페이의 논문으로 이미
  증명되었다.
  
# 만약 NP ≠ P 라면 어떤 문제가 주어졌을때 P인지 아닌지 판단하는 방법

- ???

# references

- [SNUON_컴퓨터과학이 여는 세계_10.1 P클래스와 NP클래스 문제의 개념_이광근](https://www.youtube.com/watch?v=SW0fRQQYkdA&index=34&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_10.2 NP클래스 문제의 예_이광근](https://www.youtube.com/watch?v=6rmJb_6Vx18&index=34&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH#t=5.577024)
- [SNUON_컴퓨터과학이 여는 세계_10.3 NP완전문제의 개념_이광근](https://www.youtube.com/watch?v=J4d2T7XnOT4&index=36&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_11.1 건너풀기의 개념과 어려운 문제 판별법_이광근](https://www.youtube.com/watch?v=OBcg0gg1rW8&index=37&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_11.2 어려운 문제 현실적으로 풀기: 통밥과 무작위_이광근](https://www.youtube.com/watch?v=Fi8C0Y_FWEQ&index=38&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH)
- [SNUON_컴퓨터과학이 여는 세계_11.3 어려운 문제의 적당한 해결법_이광근](https://www.youtube.com/watch?v=ZllOMcRSXFA&list=PL0Nf1KJu6Ui7yoc9RQ2TiiYL9Z0MKoggH&index=39)
