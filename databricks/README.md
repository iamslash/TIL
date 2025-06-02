- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Features](#features)
  - [Databricks Lakehouse](#databricks-lakehouse)

-----

# Abstract

Databricks는 데이터 분석, 머신러닝, 인공지능을 위한 통합 클라우드 플랫폼입니다. Apache Spark를 기반으로 하여 대규모 데이터 처리와 분석을 효율적으로 수행할 수 있습니다.

주요 특징:
- 클라우드 기반의 확장 가능한 인프라 제공
- Apache Spark 기반의 강력한 데이터 처리 엔진
- 협업을 위한 노트북 인터페이스 제공 
- MLflow를 통한 머신러닝 워크플로우 관리
- 데이터 레이크와 데이터 웨어하우스의 장점을 결합한 레이크하우스 아키텍처
- AWS, Azure, GCP 등 주요 클라우드 플랫폼 지원

Databricks는 기업이 데이터를 활용하여 인사이트를 도출하고 데이터 기반 의사결정을 내리는데 도움을 주는 플랫폼입니다.

## 실제 사용 사례

### 1. 데이터 파이프라인 구축
```python
# 데이터 파이프라인 예시
from pyspark.sql import SparkSession

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("Data Pipeline") \
    .getOrCreate()

# 데이터 읽기
df = spark.read.format("csv") \
    .option("header", "true") \
    .load("/mnt/data/raw/sales.csv")

# 데이터 변환
transformed_df = df.select(
    "date",
    "product_id",
    "quantity",
    "price"
).withColumn("total_sales", df.quantity * df.price)

# Delta Lake에 저장
transformed_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("sales_analytics.daily_sales")
```

### 2. 머신러닝 모델 개발
```python
# MLflow를 사용한 머신러닝 예시
import mlflow
from sklearn.ensemble import RandomForestRegressor

# MLflow 실험 시작
with mlflow.start_run():
    # 모델 학습
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # 모델 평가
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # MLflow에 모델 저장
    mlflow.sklearn.log_model(model, "sales_forecast_model")
    mlflow.log_metric("mse", mse)
```

### 3. SQL 분석
```sql
-- SQL 분석 예시
SELECT 
    date_trunc('month', date) as month,
    product_category,
    sum(total_sales) as monthly_sales
FROM sales_analytics.daily_sales
GROUP BY 1, 2
ORDER BY 1, 2;
```

# Materials

- [2025년도 Databricks 입문 시리즈 | 온라인 워크샵 | youtube](https://www.youtube.com/watch?v=cKkk_9C8jSY&t=2953s&ab_channel=DatabricksKorea)
- [DataBricks Ebooks](https://www.databricks.com/resources?_sft_resource_type=ebook)
  - Ebooks for data engineerings including databricks solutions.

# Basic

## Features

Databricks는 다음과 같은 주요 기능들을 제공합니다:

### Workspace
- 사용자들이 데이터 분석, 머신러닝, 협업을 수행할 수 있는 통합 환경
- 노트북, 대시보드, 라이브러리, 실험 등 다양한 리소스 관리
- 팀 단위의 협업과 프로젝트 관리 기능 제공

### Catalog
- 데이터 자산의 중앙 저장소 역할
- 데이터셋, 테이블, 뷰 등의 메타데이터 관리
- 데이터 거버넌스와 접근 제어 기능 제공
- 데이터 계보(Lineage) 추적 기능

### Compute
- 다양한 컴퓨팅 리소스 제공
  - Interactive Clusters: 대화형 분석용
  - Job Clusters: 배치 작업용
  - SQL Warehouses: SQL 쿼리 실행용
- 자동 스케일링 기능으로 리소스 최적화
- 다양한 인스턴스 타입 지원

### 데이터 엔지니어링
- Delta Lake를 통한 데이터 레이크 관리
- ETL/ELT 파이프라인 구축
- 데이터 품질 모니터링
- 스트리밍 데이터 처리

### 데이터 사이언스 & 머신러닝
- MLflow를 통한 실험 관리
- 자동화된 머신러닝 (AutoML)
- 모델 배포 및 서빙
- 분산 학습 지원

### SQL 분석
- SQL 엔드포인트 제공
- 대시보드 및 시각화 도구
- 실시간 데이터 분석
- BI 도구 연동

### 보안 & 거버넌스
- 통합 ID 관리
- 데이터 암호화
- 감사 로깅
- 규정 준수 관리

### 통합 & 확장성
- REST API 제공
- 다양한 프로그래밍 언어 지원
- 주요 클라우드 플랫폼 통합
- 커스텀 애플리케이션 개발 지원

## Databricks Lakehouse

Databricks Lakehouse는 데이터 레이크와 데이터 웨어하우스의 장점을 결합한 레이크하우스 아키텍처입니다. 이를 통해 데이터를 효율적으로 저장하고 분석할 수 있습니다.

## Catalog

Databricks Catalog는 데이터 자산을 관리하는 계층적 구조를 제공합니다.

### 계층 구조
- Catalog > Schema(Database) > Table/View/Volume
- Metastore: 여러 Catalog들의 최상위 컨테이너
- Catalog: Schema들의 논리적 그룹
- Schema: 테이블과 뷰들의 네임스페이스 
- Table: 구조화된 데이터를 저장
- View: 테이블 데이터의 가상 뷰
- Volume: 비정형 데이터를 저장하는 공간

### 주요 특징
- Unity Catalog를 통한 중앙집중식 거버넌스
- 세분화된 접근 제어(ACL)
- 데이터 공유 및 협업 지원
- 메타데이터 버전 관리
- 데이터 계보 추적

### DBFS

DBFS(Databricks File System)는 Databricks의 분산 파일 시스템입니다. 클러스터의 모든 노드에서 접근할 수 있는 분산 저장소로, HDFS와 클라우드 스토리지(S3, Azure Blob Storage 등)를 추상화하여 일관된 파일 시스템 인터페이스를 제공합니다.

DBFS는 영구 스토리지와 임시 스토리지를 모두 지원하며, 클러스터의 수명주기와 관계없이 데이터를 유지할 수 있습니다. 클라우드 스토리지를 DBFS에 마운트하면 보안 자격 증명을 안전하게 관리하면서 여러 작업공간에서 동일한 스토리지에 쉽게 접근할 수 있습니다.

주로 데이터 파일의 저장 및 공유, 노트북과 라이브러리 관리, 임시 계산 결과 저장, 머신러닝 모델 아티팩트 저장 등에 활용됩니다. DBFS를 통해 사용자는 복잡한 분산 스토리지 시스템을 의식하지 않고도 간단하게 파일을 다룰 수 있습니다.

AWS S3 마운트하기

```python
# 버킷 마운트
dbutils.fs.mount(
  source = "s3://your-bucket-name",
  mount_point = "/mnt/your-bucket-name",
  extra_configs = {"fs.s3a.access.key": "your-access-key", "fs.s3a.secret.key": "your-secret-key"}
)
```

## Data Lake

데이터 레이크하우스는 데이터 레이크와 데이터 웨어하우스의 장점을 결합한 현대적인 데이터 아키텍처입니다. 데이터 레이크의 유연한 스토리지와 데이터 웨어하우스의 강력한 분석 기능을 모두 제공합니다.

주요 특징으로는:
- ACID 트랜잭션 지원으로 데이터 일관성 보장
- 스키마 적용 및 강제로 데이터 품질 관리
- 비정형 데이터와 정형 데이터를 함께 저장
- 실시간 분석과 배치 처리 모두 지원
- 오픈 포맷(Delta Lake, Iceberg, Hudi 등) 사용으로 벤더 종속성 제거

이는 단순한 프로토콜이 아닌, 데이터를 저장하고 관리하는 종합적인 아키텍처 패턴입니다. 데이터 레이크하우스는 기존 데이터 레이크의 확장성과 비용 효율성을 유지하면서, 데이터 웨어하우스의 데이터 품질과 성능을 제공하는 것이 목표입니다.

## Delta Lake

Delta Lake는 데이터 레이크 프로토콜을 구현한 오픈 소스 프로젝트입니다. 데이터 레이크의 유연성과 데이터 웨어하우스의 강력한 분석 기능을 결합한 현대적인 데이터 관리 솔루션입니다.

### 주요 기능
- ACID 트랜잭션: 데이터 일관성 보장
- 스키마 적용: 데이터 품질 관리
- 타임 트래블: 과거 데이터 버전 조회
- 업서트/머지: 효율적인 데이터 업데이트
- 스트리밍 처리: 실시간 데이터 처리

### 사용 예시
```python
# Delta Lake 테이블 생성
spark.sql("""
CREATE TABLE IF NOT EXISTS sales_analytics.daily_sales (
    date DATE,
    product_id STRING,
    quantity INT,
    price DECIMAL(10,2),
    total_sales DECIMAL(10,2)
) USING DELTA
LOCATION '/mnt/data/processed/daily_sales'
""")

# 데이터 업데이트
spark.sql("""
MERGE INTO sales_analytics.daily_sales target
USING updates source
ON target.date = source.date AND target.product_id = source.product_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

# 과거 데이터 조회
spark.sql("""
SELECT * FROM sales_analytics.daily_sales
VERSION AS OF '2024-01-01'
""")
```



