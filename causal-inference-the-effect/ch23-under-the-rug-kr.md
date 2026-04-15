# Chapter 23: 카펫 밑으로 (Under the Rug)

> **핵심**: 이 책에서 다루지 않았지만 알아야 할 주제들

---

## 23.1 다루지 않은 주제들

### 표준오차의 세부 사항

- 이분산성 보정 (Heteroskedasticity-robust SE)
- 클러스터 표준오차의 적절한 수준 선택
- 부트스트랩 표준오차

### 다중 검정 문제 (Multiple Testing)

```
100개 변수를 테스트하면 → 5개는 우연히 유의 (p < 0.05)
→ Bonferroni 보정: p값 기준을 0.05/100 = 0.0005로 낮춤
→ False Discovery Rate (FDR) 보정
```

### 통계적 검정력 (Statistical Power)

```
검정력 = 진짜 효과가 있을 때, 유의하게 나올 확률

낮은 검정력 → 진짜 효과를 놓침 (거짓 음성)
→ 사전 검정력 분석으로 필요한 표본 크기 결정
```

### 패널 데이터 방법

- 동적 패널 모형 (Arellano-Bond)
- 이질적 기울기 모형

### 인과적 기계학습

- Causal Forests
- BART (Bayesian Additive Regression Trees)
- 이질적 처치 효과 추정

---

## 23.2 이 책 이후

| 다음 단계 | 추천 자료 |
|----------|----------|
| 계량경제학 이론 심화 | Wooldridge, *Introductory Econometrics* |
| 인과추론 심화 | Angrist & Pischke, *Mostly Harmless Econometrics* |
| 인과적 ML | Athey & Imbens 논문들 |
| 베이지안 인과추론 | McElreath, *Statistical Rethinking* |
| DAG/구조적 인과모형 | Pearl, *The Book of Why* |

---

## 최종 요약: 이 책 전체의 핵심

```
1. 좋은 연구 질문을 세운다 (Ch1-2)
2. 데이터와 관계를 이해한다 (Ch3-4)
3. DGP를 파악하고 인과 다이어그램을 그린다 (Ch5-7)
4. 경로를 분석하고 식별 전략을 세운다 (Ch8-11)
5. 적절한 도구를 선택하고 적용한다 (Ch12-22)

핵심 도구:
  - 회귀: 백도어를 직접 닫기 (Ch13)
  - 매칭: 비슷한 사람끼리 비교 (Ch14)
  - 고정효과: 같은 사람의 변화 비교 (Ch16)
  - DiD: 처치/대조 × 전/후 비교 (Ch18)
  - IV: 외생적 변동만 사용 (Ch19)
  - RD: 기준점 근처 비교 (Ch20)
```
