# abstract

- 개발자가 알아야할 통계지식에 대해 적는다.

# material

- [Think Stats 프로그래머를 위한 통계 및 데이터 분석 방법]()
  - [code](https://github.com/AllenDowney/ThinkStats2)

# terms

- 일화적증거 anecdotal evidence
  - 공개되지 않고, 일반적으로 개인적 데이터에 바탕을 둔 보고
- 응답자그룹 cohort
- 확률질량함수 Probability Mass Function, PMF
  - 정규화된 히스토그램
  - 각 값들을 확률로 변환하는 함수를 의미한다.

```python
n = float(len(t))
pmf = {}
for x, freq in hist.items():
  pmf[x] = freq / n
```
