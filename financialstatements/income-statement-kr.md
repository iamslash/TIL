# 손익계산서 핵심 용어

## 전체 구조

```
Revenue (매출)
 - COGS (매출원가)
 ─────────────────────
 = Gross Profit (매출총이익)
 - SG&A (판관비: 인건비, 마케팅, 임대료 등)
 - R&D (연구개발비)
 ─────────────────────
 = EBITDA (에비타)
 - D&A (감가상각비 + 무형자산상각비)
 ─────────────────────
 = Operating Income (영업이익)
 - Interest (이자비용)
 ─────────────────────
 = EBT (세전이익)
 - Tax (법인세)
 ─────────────────────
 = Net Income (순이익)
```

## 용어 정리

### Revenue (매출)

회사가 제품이나 서비스를 판매하여 벌어들인 총 금액이다. Top Line이라고도
부른다. 모든 비용을 빼기 전의 금액이므로 회사의 **규모와 성장성**을 나타낸다.

### COGS (Cost of Goods Sold, 매출원가)

제품/서비스를 만들거나 제공하는 데 직접 들어간 비용이다.

- 제조업: 원재료, 공장 인건비, 생산 설비 비용
- SaaS: 서버/클라우드 비용, 고객 지원 인건비
- 커머스: 상품 구매 원가, 물류비

### Gross Profit (매출총이익)

```
Gross Profit = Revenue - COGS
```

매출에서 직접 원가만 뺀 것이다. 제품/서비스 자체의 수익성을 보여준다.
Gross Profit이 낮으면 아무리 많이 팔아도 돈을 벌기 어렵다.

### SG&A (Selling, General & Administrative, 판매관리비)

제품을 직접 만드는 데는 들지 않지만 회사를 운영하는 데 필요한 비용이다.

- 인건비 (영업, 관리, 경영진)
- 마케팅비
- 사무실 임대료
- 법무, 회계 비용

### R&D (Research & Development, 연구개발비)

신제품이나 기술 개발에 투자하는 비용이다. 테크 기업에서는 SG&A와 함께 가장
큰 비중을 차지한다.

### EBITDA (Earnings Before Interest, Taxes, Depreciation and Amortization)

이자, 세금, 감가상각비를 빼기 전의 영업이익이다. **현금 창출 능력**에 가까운
지표이다.

```
EBITDA = Revenue - COGS - SG&A - R&D
       = Operating Income + D&A
```

감가상각비는 실제 현금이 나가지 않는 회계상 비용이므로, EBITDA는 "사업
운영으로 실제 벌어들인 현금"에 더 가깝다. 설비 투자 규모가 다른 기업끼리
비교할 때 유용하다.

### D&A (Depreciation & Amortization, 감가상각비)

| 구분 | 대상 | 예시 |
|------|------|------|
| **Depreciation** (감가상각) | 유형자산 | 서버, 건물, 차량 |
| **Amortization** (무형자산상각) | 무형자산 | 특허, 소프트웨어, 인수한 기술 |

자산을 구매할 때 한번에 비용 처리하지 않고, 자산의 수명에 걸쳐 나누어
비용으로 인식한다.

```
예: 서버를 100억에 구매, 내용연수 5년
  → 매년 20억씩 감가상각 (현금은 구매 시점에 이미 지출)
```

### Operating Income (영업이익, EBIT)

```
Operating Income = EBITDA - D&A
```

본업에서 벌어들인 이익이다. 감가상각을 포함하므로 EBITDA보다 항상 작거나
같다. 손익계산서의 공식 항목이다.

### Interest Expense (이자비용)

차입금에 대한 이자 비용이다. 부채가 많은 회사일수록 크다.

### EBT (Earnings Before Tax, 세전이익)

```
EBT = Operating Income - Interest Expense
```

### Tax (법인세)

국가에 납부하는 세금이다. 국가별, 기업별로 실효세율이 다르다.

### Net Income (순이익)

```
Net Income = EBT - Tax
```

모든 비용을 제한 최종 이익이다. Bottom Line이라고도 부른다. 주주에게 배당을
지급하거나 사업에 재투자하는 원천이 된다.

### EPS (Earnings Per Share, 주당순이익)

```
EPS = Net Income ÷ 발행주식수
```

주식 1주당 벌어들인 순이익이다. 주가와 직접 비교되므로 투자자가 가장 많이
보는 지표 중 하나이다.

## 크기 관계

```
Revenue ≥ Gross Profit ≥ EBITDA ≥ Operating Income ≥ EBT ≥ Net Income
```

항상 위에서 아래로 갈수록 작아진다 (비용을 계속 빼므로).
