# Basic

## Features

Delta Lake는 Databricks에서 개발한 오픈 소스 **저장소 레이어**로, 데이터 레이크의
확장성과 낮은 비용의 이점뿐만 아니라 다양한 데이터 파이프라인에서 요구되는 높은
신뢰성과 성능을 제공합니다. 이는 Apache Spark와 호환되며 현재 데이터 레이크
구조(예: Parquet, CSV, JSON 등)로 저장되어 있는 빅데이터에 걸쳐 기능을
확장합니다. 데이터 엔지니어링과 데이터 사이언스를 지원하며, 결함
허용(fault-tolerance), 스트리밍 및 배치 처리, ACID 데이터베이스의 트랜잭션
지원과 같은 다양한 기능을 제공합니다.

Delta Lake의 주요 특징은 다음과 같습니다:

- ACID 트랜잭션 지원: 원자성(Atomicity), 일관성(Consistency), 고립성(Isolation),
  지속성(Durability)를 통해 데이터 안정성과 복원력(recoverability)을 보장합니다.
- 확장 가능한 메타 데이터 처리: 빅데이터 처리에 필요한 대규모 구조의 메타
- 시간여행(Time Travel): 이전 버전의 데이터를 쿼리하거나 다시 생성할 수 있다는
  의미에서 Delta Lake는 시간여행 기능을 제공하여 데이터 변화를 추적할 수
  있습니다. 데이터를 빠르게 처리할 수 있습니다.
- Streaming과 배치의 통합: Databricks에서 제공하는 구조적 스트리밍 기능과
  호환되어, 처리를 간소화하고 속도를 높입니다.
- 체크포인트 및 파티셔닝: 데이터를 세분화하여 처리 과정을 최적화하고, 효율적인
  업데이트 작업을 지원합니다.

Delta Lake는 기존 데이터 레이크 관리를 통합하고 대규모 데이터 파이프라인을
간소화하는 데 도움이 되는 기술입니다.

## Delta Lake ACID Transaction Example

다음은 Delta Lake에서 PySpark를 사용하는 경우 ACID 트랜잭션을 활용하는 간단한 예제입니다.

1. 환경 설정 및 Delta 테이블 생성:

```py
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

# PySpark 및 Delta Lake 설정
spark = SparkSession.builder \
    .appName("Delta Lake ACID Transaction Example") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# 샘플 데이터 생성 및 Delta 레이크 테이블 저장
data = spark.createDataFrame([(1, "A"), (2, "B"), (3, "C")], ["id", "value"])
data.write.format("delta").mode("overwrite").save("delta-lake-table")
```

2. 트랜잭션이 허용되기 전 원본 데이터:

```py
# 원본 데이터 테이블 조회
original_data = spark.read.format("delta").load("delta-lake-table")
original_data.show()
```

3. ACID 트랜잭션을 사용하여 테이블 업데이트:

```py
# PySpark와 Delta Lake을 이용한 테이블 업데이트
def update_func(id, value):
    if id == 1:
        return "Z"
    return value

# 작업을 수행하려는 Delta 테이블 정보 로드
delta_table = DeltaTable.forPath(spark, "delta-lake-table")

# ACID 트랜잭션을 이용한 업데이트 작업 진행
(
    delta_table.update(
        condition="id == 1",
        set={"value": f"UPDATE(id, value)"}),
)

# 업데이트 된 테이블 조회
updated_data = spark.read.format("delta").load("delta-lake-table")
updated_data.show()
```

예제에서는 먼저 SparkSession을 사용하여 PySpark에서 Delta Lake를 설정하고 예제
데이터를 생성하고 저장합니다. 그런 다음 원본 데이터를 조회하고, ACID 트랜잭션을
사용하여 테이블을 업데이트합니다. 결과적으로 업데이트 된 테이블이 출력됩니다.

이 예제는 간단하지만 어떻게 Delta Lake를 사용하여 데이터를 안전하게 업데이트할
수 있는지 보여줍니다. 이 업데이트 과정은 원자성을 보장하므로, 여러 작업자가
동시에 같은 데이터에 접근하거나 변경하더라도 트랜잭션이 안전하게 처리됩니다.

## Delta Lake Time Travel Example

다음은 Delta Lake에서 시간 여행(time travel) 기능을 사용한 예제입니다. 이
예제에서는 PySpark로 만들어진 간단한 데이터 프레임을 활용해 Delta 테이블에
데이터를 저장하고, 현재 버전과 과거 버전의 데이터 간 차이를 비교합니다.

```py
from pyspark.sql import SparkSession

# PySpark 및 Delta Lake 설정
spark = SparkSession.builder \
    .appName("Delta Lake Time Travel Example") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# 샘플 데이터 생성 및 Delta 레이크 테이블 저장
path = "delta-lake-time-travel-table"
data = spark.createDataFrame([(1, "A"), (2, "B"), (3, "C")], ["id", "value"])
data.write.format("delta").mode("overwrite").save(path)

# 데이터 추가: Delta 테이블에 새로운 데이터 추가
new_data = spark.createDataFrame([(4, "D"), (5, "E")], ["id", "value"])
new_data.write.format("delta").mode("append").save(path)

# 현재 버전 조회
current_data = spark.read.format("delta").load(path)
print("Current version:")
current_data.show()

# 특정 버전 조회: data 'versionAsOf' 옵션 사용해 이전 버전 조회
previous_version_data = spark.read.format("delta").option("versionAsOf", 0).load(path)
print("Previous version (version 0):")
previous_version_data.show()
```

이 예제에서는 우선 PySpark에서 Delta Lake를 설정하고 샘플 데이터를 생성한 후
저장합니다. 그 다음 샘플 데이터에 새로운 데이터를 추가하여 Delta 테이블에
저장합니다. 현재 버전의 Delta 테이블을 조회한 후, 'versionAsOf' 옵션을 사용하여
이전 버전의 데이터를 불러와 조회합니다.

시간 여행(time travel) 기능을 이용하면 데이터의 여러 버전에 대한 조회 및 데이터
변화 추적이 가능해져 여러 시나리오에서 유용하게 활용할 수 있습니다.
