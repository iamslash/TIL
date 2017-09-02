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
- pmf mean

```latex
\mu = \sum_{i} p_{i} x_{i}
```

![](pmf_mean.png)

- pmf variance

```latex
\sigma^{2} = \sum_{i} p_{i} (x_{i} - \mu)^{2} 

```

![](pmf_var.png)

- 극단값 Outlier
  - 중심경향에서 멀리 떨어져있는 이상값, 특이값

- 상대위험도 ralative risk
  - 두 분포의 차이를 측정할때 쓰는 두 확률의 비율
  - 첫 아이가 출산 예정일 보다 일찍 태어날 확률은 18.2%이다. 첫아이
    외에 다른 아이가 일찍 태어날 확률은 16.8%이다. 이때 상대 위험도는
    1.08이다. 이 것은 첫아이가 출산 예정일보다 일찍 태어날 확률이
    8% 이상된다는 의미이다.

- 최빈값 mode
  - 표본에서 빈도수가 가장 높은 값

- 백분위수 percentile

```python
def PercentileRank(scores, your_score):
  count = 0
  for score in scores:
    if score <= your_score:
      count += 1
  percentile_rank = 100.0 * count / len(scores)
  return percentile_rank

def Percentile(scores, percentile_rank):
  scores.sort()
  for score in scores:
    if PercentileRank(scores, score) >= percentile_rank:
      return score
```

- 누적분포함수 Cumulative Distribution Function, CDF
  - 분포 내 각 값들을 분포내의 백분위 순위로 매핑시키는 함수이다.

```python
def Cdf(t, x):
  count = 0.0
  for value in t:
    if value <= x:
      count += 1.0
  prob = count / len(t)
  return prob
```