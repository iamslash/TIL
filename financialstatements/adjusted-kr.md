# Adjusted 지표 (조정 지표)

## 개요

Adjusted(조정) 지표는 일회성/비현금 항목을 제거하여 **"정상적 영업 활동만의
실적"**을 보여주는 지표이다. 회계 기준(GAAP)에 따른 수치와 구분하여,
회사의 반복적인 영업 성과를 파악하는 데 사용한다.

```
GAAP 지표:     회계 규칙에 따른 공식 수치
Adjusted 지표: 일회성/비현금 항목을 제거한 수치

Adjusted EBITDA ≥ EBITDA (항상)
Adjusted Operating Income ≥ Operating Income (항상)
```

## Adjusted EBITDA

EBITDA에서 일회성 비용과 비현금 비용을 더해서(제외하여) 계산한다.

```
Adjusted EBITDA = EBITDA + 제외 항목들
```

## Adjusted Operating Income

영업이익에서 일회성 비용을 더해서(제외하여) 계산한다. 감가상각은 이미 반영된
상태에서 일회성 항목만 제거한다.

```
Adjusted Operating Income = Operating Income + 일회성 비용
```

## 주로 제외하는 항목

### SBC (Stock-Based Compensation, 주식보상비)

직원에게 스톡옵션이나 RSU를 지급하는 것의 회계상 비용이다.

- **현금 지출이 없다** (주식으로 보상)
- 테크 기업에서 매우 큰 비중을 차지한다
- 매 분기 반복 발생하지만, 현금 흐름에 영향이 없으므로 Adjusted에서 제외

```
예: 분기 SBC가 100억인 테크 기업
  EBITDA:          200억
  Adjusted EBITDA: 300억  (SBC 100억 제외)
```

> 주의: SBC를 제외하면 실적이 좋아 보이지만, 실제로는 기존 주주의 지분이
> 희석(dilution)된다. SBC가 큰 기업은 Adjusted 지표를 과신하면 안 된다.

### 구조조정 비용 (Restructuring Charges)

대규모 해고, 사업부 폐쇄, 조직 개편 시 발생하는 비용이다.

- 일회성으로 발생
- 퇴직금, 리스 해지 비용, 자산 처분 손실 등

### 소송 비용 / 합의금

법적 분쟁으로 인한 비용이다.

- 대규모 합의금, 벌금
- 정상 영업과 무관한 일회성 비용

### M&A 관련 비용

인수합병 과정에서 발생하는 비용이다.

- 자문료 (투자은행, 법무법인)
- 통합 비용 (시스템 통합, 인력 조정)
- 인수 관련 무형자산 상각비

### 감손 처리 (Impairment)

자산 가치가 급격히 하락했을 때 장부가치를 한꺼번에 낮추는 것이다.

- 영업권(Goodwill) 감손
- 투자 자산 평가 손실

## 예시

```
                              GAAP      조정 항목      Adjusted
─────────────────────────────────────────────────────────────
Revenue                      1,000억                   1,000억
COGS                          -400억                    -400억
Gross Profit                   600억                     600억
SG&A                          -200억    SBC +50억       -150억
R&D                           -150억    SBC +30억       -120억
구조조정 비용                   -30억    구조조정 +30억       0
─────────────────────────────────────────────────────────────
EBITDA                         220억                     330억
D&A                            -50억                     -50억
─────────────────────────────────────────────────────────────
Operating Income               170억                     280억
```

| 지표 | GAAP | Adjusted | 차이 |
|------|------|----------|------|
| EBITDA | 220억 | 330억 | SBC 80억 + 구조조정 30억 |
| Operating Income | 170억 | 280억 | 동일 |

## GAAP vs Adjusted 를 볼 때 주의점

| 관점 | 설명 |
|------|------|
| Adjusted가 항상 높다 | 비용을 제외하므로 당연히 좋아 보인다 |
| 회사마다 제외 항목이 다르다 | A사와 B사의 Adjusted EBITDA를 단순 비교하면 안 된다 |
| SBC는 실질적 비용이다 | 현금 지출은 없지만 주주 지분이 희석된다 |
| "일회성"이 매년 반복되면 | 진짜 일회성인지 의심해야 한다 |
| 투자자 입장에서 | GAAP과 Adjusted 모두 보고, 차이가 큰 이유를 확인해야 한다 |

## 어닝콜에서 자주 보는 표현

| 표현 | 의미 |
|------|------|
| "Adjusted EBITDA margin expanded" | 조정 EBITDA 마진이 개선됨 |
| "Excluding SBC and one-time charges" | SBC와 일회성 비용을 제외하면 |
| "On a non-GAAP basis" | GAAP 기준이 아닌 조정 기준으로 |
| "GAAP to non-GAAP reconciliation" | GAAP과 Adjusted 간 차이 조정표 |
