- [Abstract](#abstract)
- [Architectures](#architectures)
- [Feast Usages](#feast-usages)
- [Feast Registry Schema](#feast-registry-schema)
- [`feast init` Deep Dive](#feast-init-deep-dive)
- [`feast apply` Deep Dive](#feast-apply-deep-dive)
- [`feast materialize` Deep Dive](#feast-materialize-deep-dive)
- [Example - Training Models, Predcting Models](#example---training-models-predcting-models)
- [Why Accessing Through Feast Instead of Online, Offline Store Directly](#why-accessing-through-feast-instead-of-online-offline-store-directly)
- [Why One Way Synchronization](#why-one-way-synchronization)

---

# Abstract

Feast는 머신러닝을 위한 오픈소스 feature store로, 모델 학습과 온라인 추론을 위한 분석 데이터를 프로덕션 환경에서 관리하고 제공하는 도구입니다.

# Architectures

Feast의 아키텍처를 다음과 같이 설명하겠습니다:

```
[Feature Repository]
    │
    ├── feature_store.yaml (설정 파일)
    │
    ├── feature_views.py (Feature 정의)
    │
    └── entities.py (Entity 정의)
         │
         ▼
[Registry]
    │
    ├── Feature Views (메타데이터)
    │
    ├── Entities (메타데이터)
    │
    ├── Data Sources (메타데이터)
    │
    └── Feature Services (메타데이터)
         │
         ▼
[Offline Store] ◄────────┐
    │                    │
    ├── Historical Data  │
    │                    │
    └── Training Data    │
         │               │
         ▼               │
[Online Store]           │
    │                    │
    ├── Feature Values   │
    │                    │
    └── Real-time Data   │
         │               │
         ▼               │
[Model Serving]          │
    │                    │
    ├── Training         │
    │   └── Offline Features
    │
    └── Inference
        └── Online Features
```

주요 컴포넌트 설명:

1. **Feature Repository**
   - Feature 정의와 설정을 포함하는 코드 저장소
   - `feature_store.yaml`: Registry, Online/Offline Store 설정
   - `feature_views.py`: Feature 정의
   - `entities.py`: Entity 정의

2. **Registry**
   - Feature Store의 메타데이터 저장소
   - Feature Views, Entities, Data Sources, Feature Services의 정의 저장
   - 변경 이력 관리
   - 프로젝트별 설정 관리

3. **Offline Store**
   - Historical 데이터 저장
   - Model training을 위한 feature 데이터 제공
   - Batch processing 지원
   - BigQuery, Snowflake, Redshift 등 지원

4. **Online Store**
   - Real-time feature serving을 위한 데이터 저장
   - Low-latency 접근 지원
   - Redis, DynamoDB, SQLite 등 지원
   - Materialized features 저장

5. **Model Serving**
   - Training: Offline Store에서 historical features 사용
   - Inference: Online Store에서 real-time features 사용

데이터 흐름:
1. Feature Repository에서 feature 정의
2. Registry에 메타데이터 등록
3. Offline Store에서 historical 데이터 처리
4. Online Store에 materialized features 저장
5. Model Serving에서 features 사용

# Feast Usages

1. **기본 설정 및 초기화**:
   - `feast init` - 새로운 feature repository 생성
   - `feast apply` - feature store 배포 생성/업데이트
   - `feast configuration` - Feast 설정 표시
   - `feast version` - Feast SDK 버전 표시

2. **Feature Store 관리**:
   - `feast materialize` - offline store에서 online store로 데이터 로드
   - `feast materialize-incremental` - 증분 materialization 실행
   - `feast teardown` - 배포된 feature store 인프라 제거
   - `feast registry-dump` - 메타데이터 registry 내용 출력

3. **서버 관련**:
   - `feast serve` - Feature server 실행 (기본 포트: 6566)
   - `feast serve-transformations` - Feature transformation 서버 실행
   - `feast serve-registry` - Registry 서버 실행 (기본 포트: 6570)
   - `feast serve-offline` - Offline 서버 실행 (기본 포트: 8815)
   - `feast ui` - Feast UI 서버 실행 (기본 포트: 8888)

4. **Entity 관리**:
   - `feast entities list` - 등록된 모든 entity 목록 표시
   - `feast entities describe` - 특정 entity 상세 정보 표시

5. **Feature View 관리**:
   - `feast feature-views list` - 등록된 모든 feature view 목록 표시
   - `feast feature-views describe` - 특정 feature view 상세 정보 표시

6. **Feature Service 관리**:
   - `feast feature-services list` - 등록된 모든 feature service 목록 표시
   - `feast feature-services describe` - 특정 feature service 상세 정보 표시

7. **Data Source 관리**:
   - `feast data-sources list` - 등록된 모든 data source 목록 표시
   - `feast data-sources describe` - 특정 data source 상세 정보 표시

8. **Feature 검색 및 접근**:
   - `feast features list` - 등록된 모든 feature 목록 표시
   - `feast get-online-features` - online feature 값 조회
   - `feast get-historical-features` - historical feature 값 조회

9. **Project 관리**:
   - `feast projects list` - 등록된 모든 project 목록 표시
   - `feast projects describe` - 특정 project 상세 정보 표시

10. **권한 관리**:
    - `feast permissions list` - 등록된 모든 permission 목록 표시
    - `feast permissions describe` - 특정 permission 상세 정보 표시

11. **Saved Dataset 관리** (실험적 기능):
    - `feast saved-datasets list` - 저장된 dataset 목록 표시
    - `feast saved-datasets describe` - 특정 dataset 상세 정보 표시

각 명령어는 `--help` 옵션을 사용하여 자세한 사용법을 확인할 수 있습니다. 예: `feast serve --help`


# Feast Registry Schema

사용자의 생체 정보(biometric)를 관리하는 feature store를 예시로 registry schema를 설명하겠습니다.

1. **Entity 정의** (`entities.py`):
```python
from feast import Entity

user = Entity(
    name="user",
    join_keys=["user_id"],
    description="사용자 식별자",
)
```

2. **Feature View 정의** (`features.py`):
```python
from feast import FeatureView, FileSource, Field
from feast.types import Float32, Int64
from datetime import timedelta

# 생체 데이터 소스 정의
biometric_source = FileSource(
    path="data/user_biometrics.parquet",
    timestamp_field="event_timestamp",
)

# 생체 정보 Feature View
biometric_features = FeatureView(
    name="user_biometric_features",
    entities=[user],
    ttl=timedelta(days=90),  # 90일 동안 데이터 유지
    schema=[
        Field(name="heart_rate", dtype=Float32),
        Field(name="blood_pressure_systolic", dtype=Int64),
        Field(name="blood_pressure_diastolic", dtype=Int64),
        Field(name="body_temperature", dtype=Float32),
        Field(name="oxygen_saturation", dtype=Float32),
    ],
    source=biometric_source,
)
```

3. **Registry Schema 구조**:

```protobuf
// Entity 정의
message Entity {
    string name = 1;  // "user"
    repeated string join_keys = 2;  // ["user_id"]
    string description = 3;  // "사용자 식별자"
}

// DataSource 정의
message DataSource {
    string name = 1;  // "biometric_source"
    string path = 2;  // "data/user_biometrics.parquet"
    string timestamp_field = 3;  // "event_timestamp"
}

// FeatureView 정의
message FeatureView {
    string name = 1;  // "user_biometric_features"
    repeated string entities = 2;  // ["user"]
    int64 ttl_seconds = 3;  // 90 days in seconds
    repeated Feature features = 4;  // heart_rate, blood_pressure 등
    string source = 5;  // "biometric_source"
}

// Feature 정의
message Feature {
    string name = 1;  // "heart_rate"
    string dtype = 2;  // "Float32"
}

// FeatureService 정의
message FeatureService {
    string name = 1;  // "biometric_service"
    repeated string feature_views = 2;  // ["user_biometric_features"]
}
```

4. **데이터베이스 스키마** (SQLite 예시):
```sql
-- Entities 테이블
CREATE TABLE entities (
    entity_name VARCHAR PRIMARY KEY,
    entity_proto BINARY NOT NULL,
    last_updated_timestamp TIMESTAMP NOT NULL
);

-- Data Sources 테이블
CREATE TABLE data_sources (
    data_source_name VARCHAR PRIMARY KEY,
    data_source_proto BINARY NOT NULL,
    last_updated_timestamp TIMESTAMP NOT NULL
);

-- Feature Views 테이블
CREATE TABLE feature_views (
    feature_view_name VARCHAR PRIMARY KEY,
    feature_view_proto BINARY NOT NULL,
    last_updated_timestamp TIMESTAMP NOT NULL
);

-- Feature Services 테이블
CREATE TABLE feature_services (
    feature_service_name VARCHAR PRIMARY KEY,
    feature_service_proto BINARY NOT NULL,
    last_updated_timestamp TIMESTAMP NOT NULL
);

-- Online Store 테이블 (실제 feature 값 저장)
CREATE TABLE user_biometric_features (
    user_id VARCHAR PRIMARY KEY,
    heart_rate FLOAT,
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    body_temperature FLOAT,
    oxygen_saturation FLOAT,
    event_timestamp TIMESTAMP
);
```

5. **Feature Store 설정** (`feature_store.yaml`):
```yaml
project: biometric_demo
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
```

이 예시에서:
- `user` entity는 사용자를 식별하는 기본 단위입니다.
- `biometric_features` feature view는 사용자의 생체 정보를 저장합니다.
- Registry는 이러한 정의들을 저장하고 관리합니다.
- Online store는 실시간으로 접근 가능한 최신 생체 정보를 저장합니다.
- Offline store는 historical 데이터를 저장하여 모델 학습에 사용할 수 있습니다.

이 구조를 통해:
1. 실시간 생체 정보 모니터링
2. 건강 상태 예측 모델 학습
3. 사용자별 건강 지표 추적
4. 이상 징후 감지

등의 기능을 구현할 수 있습니다.

# `feast init` Deep Dive

`feast init` 명령어의 구현을 찾아보겠습니다:

Search files...
`feast init` 명령어의 구현과 처리 단계를 설명해드리겠습니다:

1. **명령어 정의** (`sdk/python/feast/cli/cli.py`):

```python
@cli.command("init")
@click.argument("PROJECT_DIRECTORY", required=False)
@click.option("--minimal", "-m", is_flag=True, help="Create an empty project repository")
@click.option("--template", "-t", type=click.Choice(["local", "gcp", "aws", "snowflake", "spark", "postgres", "hbase", "cassandra", "hazelcast", "ikv", "couchbase", "milvus"]), help="Specify a template for the created project", default="local")
def init_command(project_directory, minimal: bool, template: str):
    """Create a new Feast repository"""
    if not project_directory:
        project_directory = generate_project_name()
    if minimal:
        template = "minimal"
    init_repo(project_directory, template)
```

2. **처리 단계**:

a. **프로젝트 디렉토리 생성**:

-   사용자가 지정한 디렉토리 이름으로 새 디렉토리를 생성
-   디렉토리가 지정되지 않은 경우 자동으로 이름 생성

b. **템플릿 선택**:

-   `--template` 옵션으로 선택 가능한 템플릿:
    -   local (기본값)
    -   gcp
    -   aws
    -   snowflake
    -   spark
    -   postgres
    -   hbase
    -   cassandra
    -   hazelcast
    -   ikv
    -   couchbase
    -   milvus

c. **기본 파일 구조 생성**:

```
project_directory/
├── data/
│   └── driver_stats.parquet
├── feature_repo/
│   ├── __init__.py
│   ├── entities.py
│   ├── features.py
│   └── feature_store.yaml
└── requirements.txt
```

d. **feature_store.yaml 생성**:

```yaml
project: project_name
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
```

e. **샘플 데이터 생성**:

-   `data/driver_stats.parquet` 파일 생성
-   샘플 feature view와 entity 정의

3. **주요 기능**:

-   새로운 feature repository 초기화
-   기본 설정 파일 생성
-   샘플 feature view와 entity 정의
-   로컬 개발 환경 설정
-   다양한 클라우드 제공자 템플릿 지원

4. **사용 예시**:

```bash
# 기본 로컬 템플릿으로 생성
feast init my_project

# 최소한의 템플릿으로 생성
feast init my_project --minimal

# 특정 클라우드 제공자 템플릿으로 생성
feast init my_project --template gcp
```

이 명령어는 Feast 프로젝트를 시작하기 위한 기본 구조를 생성하며, 사용자가 선택한 템플릿에 따라 적절한 설정과 샘플 코드를 제공합니다.

# `feast apply` Deep Dive

`feast apply` 명령어의 처리 과정을 실제 코드를 기반으로 단계별로 설명하겠습니다:

1. **명령어 실행** (`sdk/python/feast/cli/cli.py`):
```python
@cli.command("apply")
def apply_total_command(ctx: click.Context, skip_source_validation: bool):
    """Create or update a feature store deployment"""
    repo = ctx.obj["CHDIR"]
    fs_yaml_file = ctx.obj["FS_YAML_FILE"]
    repo_config = load_repo_config(repo, fs_yaml_file)
    apply_total(repo_config, repo, skip_source_validation)
```

1. **Feature Store 초기화 및 적용** (`sdk/python/feast/feature_store.py`):
```python
def apply(self, objects, objects_to_delete=None, partial=True):
    # 1. 객체 분류
    projects_to_update = [ob for ob in objects if isinstance(ob, Project)]
    entities_to_update = [ob for ob in objects if isinstance(ob, Entity)]
    views_to_update = [ob for ob in objects if isinstance(ob, FeatureView)]
    
    # 2. Feature View 검증
    self._validate_all_feature_views(views_to_update, odfvs_to_update, sfvs_to_update)
    
    # 3. Registry에 객체 등록
    for project in projects_to_update:
        self._registry.apply_project(project, commit=False)
    for ds in data_sources_to_update:
        self._registry.apply_data_source(ds, project=self.project, commit=False)
    for view in views_to_update:
        self._registry.apply_feature_view(view, project=self.project, commit=False)
    
    # 4. 인프라 업데이트
    self._get_provider().update_infra(
        project=self.project,
        tables_to_delete=tables_to_delete,
        tables_to_keep=tables_to_keep,
        entities_to_delete=entities_to_delete,
        entities_to_keep=entities_to_update,
        partial=partial,
    )
    
    # 5. Registry 커밋
    self._registry.commit()
```

1. **Registry 업데이트** (`sdk/python/feast/infra/registry/file.py`):
```python
def update_registry_proto(self, registry_proto: RegistryProto):
    self._write_registry(registry_proto)

def _write_registry(self, registry_proto: RegistryProto):
    registry_proto.version_id = str(uuid.uuid4())
    registry_proto.last_updated.FromDatetime(_utc_now())
    file_dir = self._filepath.parent
    file_dir.mkdir(exist_ok=True)
    with open(self._filepath, mode="wb", buffering=0) as f:
        f.write(registry_proto.SerializeToString())
```

주요 처리 단계:
1. Feature repository의 Python 파일들을 스캔하여 feature 정의를 로드
2. Feature 정의의 유효성 검사 (중복 이름, 필수 필드 등)
3. Registry에 feature 메타데이터 동기화
4. Online store와 offline store에 필요한 테이블 생성
5. Registry 변경사항 커밋

이 과정을 통해 feature store가 실제로 사용 가능한 상태가 됩니다.

# `feast materialize` Deep Dive

`feast materialize` 명령어의 처리 과정을 실제 코드를 기반으로 단계별로 설명하겠습니다:

Search files...
`feast materialize` 명령어의 처리 과정을 실제 코드를 기반으로 단계별로 설명하겠습니다:

1. **명령어 실행** (`sdk/python/feast/feature_store.py`):
```python
def materialize(
    self,
    start_date: datetime,
    end_date: datetime,
    feature_views: Optional[List[str]] = None,
) -> None:
    """
    Materialize data from the offline store into the online store.
    """
```

2. **Compute Engine 선택 및 실행** (`sdk/python/feast/infra/compute_engines/base.py`):
```python
def materialize(
    self,
    registry: BaseRegistry,
    tasks: Union[MaterializationTask, List[MaterializationTask]],
    **kwargs,
) -> List[MaterializationJob]:
    if isinstance(tasks, MaterializationTask):
        tasks = [tasks]
    return [self._materialize_one(registry, task, **kwargs) for task in tasks]
```

3. **Local Compute Engine에서의 Materialization** (`sdk/python/feast/infra/compute_engines/local/compute.py`):
```python
def _materialize_one(
    self, registry: BaseRegistry, task: MaterializationTask, **kwargs
) -> LocalMaterializationJob:
    job_id = f"{task.feature_view.name}-{task.start_time}-{task.end_time}"
    context = self.get_execution_context(registry, task)
    backend = self._get_backend(context)

    try:
        builder = LocalFeatureBuilder(task, backend=backend)
        plan = builder.build()
        plan.execute(context)
        return LocalMaterializationJob(
            job_id=job_id,
            status=MaterializationJobStatus.SUCCEEDED,
        )
    except Exception as e:
        return LocalMaterializationJob(
            job_id=job_id,
            status=MaterializationJobStatus.ERROR,
            error=e,
        )
```

4. **Snowflake Compute Engine에서의 Materialization** (`sdk/python/feast/infra/compute_engines/snowflake/snowflake_engine.py`):
```python
def _materialize_one(
    self,
    registry: BaseRegistry,
    task: MaterializationTask,
    **kwargs,
):
    feature_view = task.feature_view
    start_date = task.start_time
    end_date = task.end_time
    project = task.project

    # 1. Entity 정보 가져오기
    entities = []
    for entity_name in feature_view.entities:
        entities.append(registry.get_entity(entity_name, project))

    # 2. 컬럼 정보 가져오기
    join_key_columns, feature_name_columns, timestamp_field, created_timestamp_column = _get_column_names(feature_view, entities)

    # 3. Offline Store에서 데이터 가져오기
    offline_job = self.offline_store.pull_latest_from_table_or_query(
        config=self.repo_config,
        data_source=feature_view.batch_source,
        join_key_columns=join_key_columns,
        feature_name_columns=feature_name_columns,
        timestamp_field=timestamp_field,
        created_timestamp_column=created_timestamp_column,
        start_date=start_date,
        end_date=end_date,
    )

    # 4. Online Store에 데이터 쓰기
    if self.repo_config.online_store.type == "snowflake.online":
        self.materialize_to_snowflake_online_store(
            self.repo_config,
            fv_to_proto_sql,
            feature_view,
            project,
        )
    else:
        self.materialize_to_external_online_store(
            self.repo_config,
            fv_to_proto_sql,
            feature_view,
            pbar,
        )
```

주요 처리 단계:
1. Feature Store의 materialize 메서드 호출
2. Compute Engine 선택 (Local, Snowflake, Spark 등)
3. Feature View별로 materialization task 생성
4. Offline Store에서 데이터 읽기
5. Online Store에 데이터 쓰기
6. Materialization 상태 업데이트

이 과정을 통해 offline store의 데이터가 online store로 materialize됩니다.

# Example - Training Models, Predcting Models

생체 정보를 기반으로 한 건강 상태 예측 모델의 학습과 서빙 코드를 제공하겠습니다.

1. **모델 학습 코드** (`train_model.py`):
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from feast import FeatureStore
import joblib
from datetime import datetime, timedelta

def prepare_training_data():
    # Feature Store 초기화
    store = FeatureStore(repo_path=".")
    
    # 학습 기간 설정 (예: 최근 90일)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Entity 데이터 준비 (예: 1000명의 사용자)
    user_ids = [f"user_{i}" for i in range(1000)]
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Entity DataFrame 생성
    entity_df = pd.DataFrame([
        {"user_id": user_id, "event_timestamp": timestamp}
        for user_id in user_ids
        for timestamp in timestamps
    ])
    
    # Feature 데이터 조회
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_biometric_features:heart_rate",
            "user_biometric_features:blood_pressure_systolic",
            "user_biometric_features:blood_pressure_diastolic",
            "user_biometric_features:body_temperature",
            "user_biometric_features:oxygen_saturation"
        ]
    ).to_df()
    
    return features

def train_health_status_model():
    # 데이터 준비
    features_df = prepare_training_data()
    
    # 결측치 처리
    features_df = features_df.dropna()
    
    # 건강 상태 레이블 생성 (예시: 비정상적인 생체 지표 조합)
    features_df['health_status'] = np.where(
        (features_df['heart_rate'] > 100) |  # 높은 심박수
        (features_df['blood_pressure_systolic'] > 140) |  # 높은 수축기 혈압
        (features_df['body_temperature'] > 37.5) |  # 발열
        (features_df['oxygen_saturation'] < 95),  # 낮은 산소 포화도
        1,  # 비정상
        0   # 정상
    )
    
    # 특성과 타겟 분리
    X = features_df[[
        'heart_rate',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'body_temperature',
        'oxygen_saturation'
    ]]
    y = features_df['health_status']
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 모델 평가
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # 모델 저장
    joblib.dump(model, 'models/health_status_model.joblib')
    
    return model

if __name__ == "__main__":
    model = train_health_status_model()
```

2. **모델 서빙 코드** (`serve_model.py`):
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
import joblib
import numpy as np
from typing import List, Dict
import uvicorn

app = FastAPI(title="Health Status Prediction API")

# 모델과 Feature Store 초기화
model = joblib.load('models/health_status_model.joblib')
store = FeatureStore(repo_path=".")

class HealthPredictionRequest(BaseModel):
    user_ids: List[str]

class HealthPredictionResponse(BaseModel):
    predictions: Dict[str, Dict[str, float]]

@app.post("/predict", response_model=HealthPredictionResponse)
async def predict_health_status(request: HealthPredictionRequest):
    try:
        # Entity 데이터 준비
        entity_rows = [{"user_id": user_id} for user_id in request.user_ids]
        
        # Feature 데이터 조회
        features = store.get_online_features(
            entity_rows=entity_rows,
            features=[
                "user_biometric_features:heart_rate",
                "user_biometric_features:blood_pressure_systolic",
                "user_biometric_features:blood_pressure_diastolic",
                "user_biometric_features:body_temperature",
                "user_biometric_features:oxygen_saturation"
            ]
        ).to_dict()
        
        # 예측 수행
        predictions = {}
        for user_id in request.user_ids:
            # 사용자의 feature 데이터 추출
            user_features = np.array([[
                features['heart_rate'][i],
                features['blood_pressure_systolic'][i],
                features['blood_pressure_diastolic'][i],
                features['body_temperature'][i],
                features['oxygen_saturation'][i]
            ] for i, uid in enumerate(features['user_id']) if uid == user_id])
            
            if len(user_features) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"No features found for user {user_id}"
                )
            
            # 예측 확률 계산
            prediction_proba = model.predict_proba(user_features)[0]
            
            # 결과 저장
            predictions[user_id] = {
                "normal_probability": float(prediction_proba[0]),
                "abnormal_probability": float(prediction_proba[1])
            }
        
        return HealthPredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

3. **API 사용 예시**:
```python
import requests
import json

# API 호출 예시
def get_health_predictions(user_ids):
    url = "http://localhost:8000/predict"
    response = requests.post(
        url,
        json={"user_ids": user_ids}
    )
    return response.json()

# 사용 예시
user_ids = ["user_1", "user_2", "user_3"]
predictions = get_health_predictions(user_ids)
print(json.dumps(predictions, indent=2))
```

4. **실행 방법**:

```bash
# 모델 학습
python train_model.py

# API 서버 실행
python serve_model.py
```

이 코드는 다음과 같은 특징을 가집니다:

1. **학습 코드**:
   - Feast를 통한 historical feature 데이터 조회
   - 데이터 전처리 및 레이블 생성
   - 모델 학습 및 평가
   - 학습된 모델 저장

2. **서빙 코드**:
   - FastAPI를 사용한 REST API 구현
   - Feast를 통한 online feature 데이터 조회
   - 실시간 예측 수행
   - 에러 처리 및 응답 포맷팅

3. **데이터 흐름**:
   - Offline store → 모델 학습
   - Online store → 실시간 예측
   - Feature Store를 통한 일관된 데이터 접근

4. **확장성**:
   - 여러 사용자에 대한 배치 예측 지원
   - 에러 처리 및 로깅
   - 건강 상태 모니터링 API

이 코드를 실행하기 전에 필요한 패키지를 설치해야 합니다:
```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib requests
```

# Why Accessing Through Feast Instead of Online, Offline Store Directly

Feast를 통한 데이터 접근의 이점을 생체 정보 예시와 함께 설명하겠습니다.

1. **Feature 정의의 일관성**:

```python
# Feature View 정의 (features.py)
from feast import FeatureView, FileSource, Field
from feast.types import Float32, Int64

biometric_features = FeatureView(
    name="user_biometric_features",
    entities=[user],
    schema=[
        Field(name="heart_rate", dtype=Float32, description="심박수 (bpm)"),
        Field(name="blood_pressure_systolic", dtype=Int64, description="수축기 혈압 (mmHg)"),
        Field(name="blood_pressure_diastolic", dtype=Int64, description="이완기 혈압 (mmHg)"),
        Field(name="body_temperature", dtype=Float32, description="체온 (°C)"),
        Field(name="oxygen_saturation", dtype=Float32, description="산소 포화도 (%)")
    ],
    source=biometric_source
)
```

**직접 접근의 문제점**:
```python
# 잘못된 방법: 직접 접근
import pandas as pd

# 데이터 소스 직접 접근
df = pd.read_parquet("data/user_biometrics.parquet")

# 문제점:
# 1. Feature의 의미와 단위를 알 수 없음
# 2. 데이터 타입이 명시적으로 정의되지 않음
# 3. Feature 간의 관계를 알 수 없음
```

**Feast를 통한 접근**:
```python
# 올바른 방법: Feast를 통한 접근
from feast import FeatureStore

store = FeatureStore(repo_path=".")
features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_biometric_features:heart_rate",
        "user_biometric_features:blood_pressure_systolic"
    ]
).to_df()

# 이점:
# 1. Feature의 의미와 단위가 명확히 정의됨
# 2. 데이터 타입이 보장됨
# 3. Feature 간의 관계가 명시적으로 정의됨
```

2. **데이터 변환과 전처리**:

```python
# Feature View에 변환 로직 추가
from feast import FeatureView, FileSource, Field, ValueType
from feast.types import Float32

biometric_features = FeatureView(
    name="user_biometric_features",
    entities=[user],
    schema=[
        Field(
            name="heart_rate",
            dtype=Float32,
            description="심박수 (bpm)",
            validation=lambda x: 40 <= x <= 200  # 유효한 심박수 범위 검증
        ),
        Field(
            name="body_temperature",
            dtype=Float32,
            description="체온 (°C)",
            validation=lambda x: 35 <= x <= 42  # 유효한 체온 범위 검증
        )
    ],
    source=biometric_source
)
```

3. **시간 기반 필터링**:

```python
# 직접 접근의 문제점
import pandas as pd
from datetime import datetime, timedelta

# 잘못된 방법
df = pd.read_parquet("data/user_biometrics.parquet")
df = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] <= '2024-03-01')]
# 문제점: 시간 필터링이 일관되지 않을 수 있음

# Feast를 통한 접근
features = store.get_historical_features(
    entity_df=entity_df,
    features=["user_biometric_features:heart_rate"],
    start_date="2024-01-01",
    end_date="2024-03-01"
)
# 이점: 일관된 시간 필터링 보장
```

4. **데이터 품질 보장**:

```python
# Feature View에 데이터 품질 검증 추가
from feast import FeatureView, FileSource, Field
from feast.types import Float32

def validate_heart_rate(x: float) -> bool:
    return 40 <= x <= 200

def validate_temperature(x: float) -> bool:
    return 35 <= x <= 42

biometric_features = FeatureView(
    name="user_biometric_features",
    entities=[user],
    schema=[
        Field(
            name="heart_rate",
            dtype=Float32,
            validation=validate_heart_rate
        ),
        Field(
            name="body_temperature",
            dtype=Float32,
            validation=validate_temperature
        )
    ],
    source=biometric_source
)
```

5. **실제 사용 예시**:

```python
# 모델 학습 시
def prepare_training_data():
    store = FeatureStore(repo_path=".")
    
    # Entity 데이터 준비
    entity_df = pd.DataFrame({
        "user_id": ["user_1", "user_2", "user_3"],
        "event_timestamp": pd.date_range(start="2024-01-01", periods=3)
    })
    
    # Feature 데이터 조회
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_biometric_features:heart_rate",
            "user_biometric_features:blood_pressure_systolic",
            "user_biometric_features:body_temperature"
        ]
    ).to_df()
    
    return features

# 실시간 예측 시
def get_realtime_predictions(user_ids):
    store = FeatureStore(repo_path=".")
    
    # Online feature 조회
    features = store.get_online_features(
        entity_rows=[{"user_id": uid} for uid in user_ids],
        features=[
            "user_biometric_features:heart_rate",
            "user_biometric_features:blood_pressure_systolic",
            "user_biometric_features:body_temperature"
        ]
    ).to_dict()
    
    return features
```

6. **데이터 버전 관리**:

```python
# Feature View 버전 관리
biometric_features_v1 = FeatureView(
    name="user_biometric_features_v1",
    entities=[user],
    schema=[...],
    source=biometric_source
)

biometric_features_v2 = FeatureView(
    name="user_biometric_features_v2",
    entities=[user],
    schema=[...],  # 업데이트된 스키마
    source=biometric_source
)

# 특정 버전의 feature 사용
features = store.get_historical_features(
    entity_df=entity_df,
    features=["user_biometric_features_v1:heart_rate"]  # 특정 버전 지정
)
```

Feast를 통한 데이터 접근의 주요 이점:

1. **일관성**:
   - Feature 정의가 코드로 명시됨
   - 데이터 타입과 검증이 보장됨
   - Feature 간의 관계가 명확함

2. **데이터 품질**:
   - 자동화된 데이터 검증
   - 이상치 탐지 및 처리
   - 데이터 정합성 보장

3. **운영 효율성**:
   - 중앙화된 Feature 관리
   - 버전 관리 용이
   - 모니터링 및 추적 가능

4. **확장성**:
   - 새로운 Feature 추가 용이
   - 다양한 저장소 지원
   - 파이프라인 자동화 가능

이러한 이점들로 인해, 직접적인 저장소 접근보다 Feast를 통한 데이터 접근이 더 안전하고 효율적입니다.

# Why One Way Synchronization

Feast가 offline store에서 online store로만 동기화하는 단방향 설계를 채택한 이유를 설명하겠습니다:

1. **데이터의 특성과 용도**:
   - **Offline Store**: 
     - Historical 데이터 저장
     - 모델 학습용 데이터
     - Batch processing에 적합
     - 데이터의 정확성과 일관성이 중요
   
   - **Online Store**:
     - 실시간 서빙을 위한 최신 데이터
     - Low-latency 접근 필요
     - 실시간 업데이트가 필요한 데이터
     - 최신성과 빠른 접근이 중요

2. **데이터 파이프라인의 일반적인 흐름**:
   ```
   Raw Data → Offline Store → Feature Engineering → Online Store → Model Serving
   ```
   - Raw 데이터는 먼저 offline store에 저장
   - Feature engineering이 offline에서 수행
   - 처리된 feature가 online store로 materialize
   - Online store는 serving을 위한 최종 단계

3. **데이터 일관성 보장**:
   - Offline store는 "source of truth" 역할
   - Online store는 offline store의 subset
   - 단방향 동기화로 데이터 일관성 유지 용이
   - 양방향 동기화는 데이터 불일치 위험 증가

4. **성능과 확장성**:
   - Online store는 실시간 서빙에 최적화
   - Offline store는 대용량 데이터 처리에 최적화
   - 각각의 특성에 맞는 저장소 사용
   - 양방향 동기화는 성능 오버헤드 발생

5. **사용 사례**:
   ```python
   # Offline에서 feature 생성
   features = store.get_historical_features(
       entity_df=entity_df,
       features=[
           "user_biometric_features:heart_rate",
           "user_biometric_features:blood_pressure"
       ]
   )
   
   # Online에서 feature 서빙
   features = store.get_online_features(
       entity_rows=[{"user_id": "user_1"}],
       features=[
           "user_biometric_features:heart_rate",
           "user_biometric_features:blood_pressure"
       ]
   )
   ```

6. **데이터 품질과 검증**:
   - Offline store에서 데이터 품질 검증
   - Feature engineering 파이프라인 검증
   - Materialization 과정에서의 검증
   - Online store는 검증된 데이터만 저장

7. **비용 효율성**:
   - Offline store는 저비용 저장소 사용 가능
   - Online store는 고성능 저장소 필요
   - 단방향 동기화로 저장 비용 최적화
   - 불필요한 데이터 중복 방지

8. **운영 복잡성**:
   - 단방향 동기화로 운영 복잡성 감소
   - 데이터 파이프라인 모니터링 용이
   - 문제 발생 시 추적 용이
   - 시스템 안정성 향상

이러한 설계는 ML 시스템의 일반적인 요구사항과 데이터 파이프라인의 특성을 고려한 결과입니다. Offline store에서 online store로의 단방향 동기화는 데이터의 일관성, 성능, 운영 효율성 측면에서 최적의 선택입니다.
