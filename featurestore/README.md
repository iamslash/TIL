# Abstract

ML 추론 시 **Feature Store**가 필요한 이유는 다음과 같은 문제를 해결하고, 운영 환경에서 안정적이고 일관된 예측을 가능하게 하기 위해서입니다:

---

## 🔁 1. **Feature Consistency (특징 일관성)**

* \*\*학습(train)\*\*과 **추론(inference)** 시점에 사용하는 feature가 동일해야 모델이 정확하게 작동합니다.
* 예를 들어, 학습 시에는 `age = current_year - birth_year`로 feature를 만들었는데 추론 시에는 실수로 `age = current_year - birth_year - 1`을 사용한다면 성능이 저하되거나 이상한 예측이 나올 수 있습니다.
* **Feature Store**는 이러한 feature 정의 로직을 **중앙 집중화**하여 재사용 가능하게 하고, 동일한 로직을 학습과 추론에 모두 적용할 수 있도록 도와줍니다.

---

## ⚙️ 2. **실시간 추론을 위한 Serving 지원**

* 온라인 환경에서는 \*\*낮은 지연시간(latency)\*\*으로 feature를 조회해서 모델에 넣어야 합니다.
* Feature Store는 **online serving store**를 갖추고 있어 빠르게 feature를 조회할 수 있도록 합니다.
  예) Redis, DynamoDB 등으로 low-latency 조회

---

## 🧪 3. **실험 관리와 재현성**

* 어떤 실험에서 어떤 feature를 썼는지 추적하고 기록할 수 있어야 합니다.
* Feature Store는 feature 버전(version), 사용 이력(lineage), metadata 등을 관리해줍니다.
* 이를 통해 모델 성능이 안 좋아졌을 때 원인을 추적하거나 이전 실험을 재현할 수 있습니다.

---

## 🔁 4. **재사용성과 협업**

* 팀 단위로 feature를 정의하고 공유할 수 있어 **중복 구현을 줄이고 생산성을 높일 수 있습니다**.
* 예를 들어, 여러 모델에서 자주 쓰이는 “유저의 최근 로그인 횟수” 같은 feature를 공통으로 정의하고 재사용할 수 있습니다.

---

## 🔐 5. **데이터 품질 관리 및 모니터링**

* Feature Store는 feature의 통계정보(statistics)나 데이터 분포를 추적하고, 이상치 탐지 기능을 제공하기도 합니다.
* 이를 통해 추론 시 데이터 드리프트(data drift)나 품질 이슈를 모니터링할 수 있습니다.

---

## 정리

| 기능             | 이점            |
| -------------- | ------------- |
| Feature 정의 일관성 | 학습/추론 간 오차 제거 |
| 실시간 Feature 조회 | 온라인 추론 최적화    |
| 버전 관리          | 실험 재현, 모델 추적  |
| Feature 재사용    | 팀 생산성 향상      |
| 품질 모니터링        | 안정적인 운영 가능    |

# Products

현재 가장 유명하고 널리 사용되는 Feature Store는 다음과 같습니다. 각각의 도구는 목적, 규모, 사용 환경에 따라 강점이 다르며, 기업들이 요구하는 **실시간/배치 처리**, **스케일**, **ML 플랫폼 통합** 등의 요구사항에 따라 선택됩니다.

---

## 🔝 1. **Feast (Feature Store)**

### ✅ 특징:

* **오픈소스**로 가장 많이 쓰임
* Google Cloud, AWS, Azure 등 **클라우드 독립적**
* 온라인/오프라인 저장소 분리: Redis + BigQuery 등 조합 가능
* Python SDK, CLI, REST API 지원

### 💡 장점:

* 빠른 셋업 가능
* 유연한 아키텍처
* **Kubeflow, TFX 등과 통합 용이**

### 🔧 사용 예:

* Uber, Gojek, Shopify, Robinhood

---

## 🏢 2. **Tecton**

### ✅ 특징:

* **상업용 Feature Store**, Feast를 만든 Gojek 팀의 창업자들이 개발
* 실시간 피처 계산 및 저장을 자동화
* Snowflake, Databricks, Airflow, Kafka 등과 통합

### 💡 장점:

* 엔터프라이즈급 SLA와 성능
* 모니터링, 피처 테스트, 버저닝 등 **운영에 최적화**

### 💰 한계:

* 유료 (대규모 팀/기업 대상)

---

## 🌐 3. **Databricks Feature Store**

### ✅ 특징:

* **Databricks 환경에 최적화**
* Delta Lake 기반으로 offline feature 관리
* MLflow와 통합되어 실험, 모델 서빙까지 연결

### 💡 장점:

* Spark 기반의 대규모 처리에 적합
* Feature engineering + ML pipeline 통합 가능

---

## 🧊 4. **Amazon SageMaker Feature Store**

### ✅ 특징:

* AWS SageMaker 생태계 전용
* 실시간/배치 조회 모두 지원
* S3 + low-latency online store 구조

### 💡 장점:

* AWS 서비스 (Athena, Glue 등)와 원활한 통합
* 보안 및 IAM 설정이 AWS-native

---

## 🔷 5. **Vertex AI Feature Store (Google Cloud)**

### ✅ 특징:

* Google Cloud AI 플랫폼과 통합
* BigQuery, Dataflow와 자연스럽게 연결
* Feature drift 모니터링 제공

---

## 정리: 주요 Feature Store 비교표

| 이름                | 오픈소스 | 실시간 서빙 | 플랫폼 통합             | 특징                  |
| ----------------- | ---- | ------ | ------------------ | ------------------- |
| **Feast**         | ✅    | ✅      | Kubeflow, TFX      | 유연하고 가볍고 가장 인기      |
| **Tecton**        | ❌    | ✅      | Snowflake, Kafka 등 | 상용, 강력한 엔터프라이즈 기능   |
| **Databricks FS** | ❌    | 일부 지원  | MLflow, Delta Lake | Spark 환경 최적화        |
| **SageMaker FS**  | ❌    | ✅      | AWS 전용             | AWS 환경 통합           |
| **Vertex AI FS**  | ❌    | ✅      | GCP 전용             | Feature drift 감지 제공 |

---

