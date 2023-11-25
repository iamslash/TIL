- [Materials](#materials)
- [Basic](#basic)
  - [Run MlFlow](#run-mlflow)
  - [MlFlow](#mlflow)
  - [Mlflow and Databricks](#mlflow-and-databricks)
  - [MLflow Tracking Example](#mlflow-tracking-example)
  - [MLflow Projects Example](#mlflow-projects-example)
  - [MLflow Models Example](#mlflow-models-example)
  - [MLflow Model Registry Example](#mlflow-model-registry-example)

-----

# Materials

- [mlflow | github](https://github.com/mlflow/mlflow/)

# Basic

## Run MlFlow

- [MLflow On-Premise Deployment using Docker Compose](https://github.com/sachua/mlflow-docker-compose)

```bash
git clone https://github.com/sachua/mlflow-docker-compose.git
cd mlflow-docker-compose
docker-compose up --build
# MLFlow UI http://localhost:5000
# MinIO UI http://localhost:9000
``` 

## MlFlow

MLflow는 머신러닝 라이프사이클 관리를 위한 오픈소스 플랫폼입니다. 이 플랫폼은
머신러닝 프로젝트의 다양한 단계를 관리하고, 실험을 추적하며, 모델을 배포하는 데
도움을 줍니다. MLflow의 주요 기능은 다음과 같습니다:

- 실험 추적 (Tracking): MLflow는 머신러닝 실험의 파라미터, 코드 버전, 메트릭,
  그리고 결과를 기록하고 비교하는 데 사용됩니다. 이를 통해 실험들을 체계적으로
  관리하고 결과를 분석할 수 있습니다.
- 프로젝트 (Projects): MLflow 프로젝트는 머신러닝 코드를 재사용 가능하고 공유
  가능한 형태로 패키징하는 방법을 제공합니다. 이를 통해 다른 사용자들이 코드를
  쉽게 실행하고 결과를 재현할 수 있습니다.
- 모델 관리 (Models): 이 기능은 머신러닝 모델을 다양한 ML 프레임워크에서 사용할
  수 있는 표준 포맷으로 저장하고, 이를 배포 준비 상태로 만들어줍니다. 모델
  서빙이나 배포를 위한 다양한 플랫폼과 호환됩니다.
- 레지스트리 (Model Registry): 모델 레지스트리는 모델의 전체 라이프사이클을
  관리하는데 도움을 줍니다. 이를 통해 모델 버전 관리, 스테이지 관리 (개발,
  스테이징, 프로덕션 등) 및 모델의 협업과 거버넌스를 지원합니다.

MLflow는 머신러닝의 실험 단계부터 프로덕션 배포에 이르기까지 전 과정을 지원하며,
머신러닝 개발 및 운영의 복잡성을 줄이는 데 유용합니다. 또한, 다양한 머신러닝
프레임워크와 도구들과 호환되어 머신러닝 워크플로우를 통합하는 데 도움을 줍니다.

## Mlflow and Databricks

MLflow와 Databricks의 연동은 머신러닝 워크플로우의 효율성과 편의성을 크게
향상시킵니다. Databricks는 빅데이터 처리와 머신러닝을 위한 통합 플랫폼으로,
MLflow와 긴밀하게 통합되어 있습니다. 이러한 통합은 다음과 같은 방법으로
이루어집니다:

- 내장된 MLflow 지원: Databricks는 MLflow를 플랫폼에 내장하고 있어, 별도의
  설치나 설정 없이 MLflow의 기능을 사용할 수 있습니다. 이를 통해 사용자는
  Databricks 환경 내에서 바로 실험 추적, 모델 관리 및 레지스트리 기능을 활용할
  수 있습니다.
- 실험 추적과 관리: Databricks 환경에서 실행되는 모든 머신러닝 실험은 자동으로
  MLflow에 의해 추적됩니다. 이는 Databricks 노트북 또는 작업에서 실행되는
  코드에서 MLflow API를 사용하여 구현됩니다. 사용자는 Databricks UI를 통해 실험
  결과, 메트릭, 파라미터 등을 쉽게 볼 수 있습니다.
- 모델 서빙과 배포: Databricks에서 훈련된 MLflow 모델은 Databricks 환경 내에서
  쉽게 서빙 및 배포할 수 있습니다. 또한, Databricks는 MLflow 모델을 다양한 외부
  시스템으로 내보내는 기능도 지원합니다.
- 데이터 및 리소스 통합: Databricks는 대규모 데이터셋을 쉽게 처리할 수 있으며,
  이 데이터는 MLflow를 통해 실행되는 머신러닝 실험에 직접적으로 활용될 수
  있습니다. 또한, 클러스터 관리, 스파크 통합 등 Databricks의 리소스 관리 기능은
  MLflow 실험을 위한 강력한 백엔드를 제공합니다.
- 보안 및 거버넌스: Databricks 플랫폼의 보안 기능은 MLflow와 통합되어, 모델과
  데이터의 보안 및 거버넌스를 강화합니다.

이러한 통합을 통해 사용자는 Databricks의 강력한 데이터 처리 능력과 MLflow의
머신러닝 라이프사이클 관리 기능을 함께 활용하여 보다 효율적이고 체계적인
머신러닝 프로젝트를 수행할 수 있습니다.

## MLflow Tracking Example

이 예제에서는 scikit-learn을 사용하여 간단한 선형 회귀 모델을 훈련하고, MLflow를
사용하여 실험을 추적합니다.

```bash
pip install mlflow
vim a.py
python a.py
# mlruns directory is created.
mlflow ui
# Open browser for ui http://127.0.0.1:5000
```

```py
# a.py
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 준비
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# MLflow Tracking 서버 URI 설정 (옵션, 필요한 경우만)
# mlflow.set_tracking_uri("http://your_mlflow_server:port")

# MLflow 실험 시작
with mlflow.start_run():

    # 모델 생성 및 훈련
    model = LinearRegression()
    model.fit(X, y)

    # 예측 및 메트릭 계산
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    # 파라미터, 메트릭, 모델 로깅
    mlflow.log_param("model_type", "linear_regression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
```

## MLflow Projects Example

MLflow Projects는 머신러닝 코드를 재사용 가능하고 공유 가능한 형식으로
패키징하는 메커니즘을 제공합니다. 이를 통해 다른 사용자들이 코드를 쉽게 실행하고
결과를 재현할 수 있도록 합니다. MLflow Project는 일반적으로 두 가지 주요 파일을
포함합니다: `MLproject` 파일과 `conda.yaml` 파일 (또는 다른 환경 관리 파일).

이 예제에서는 간단한 MLflow Project를 구성합니다. 이 프로젝트는 Python
스크립트를 실행하여 선형 회귀 모델을 훈련하고 MLflow로 추적하는 작업을
수행합니다.

**프로젝트 구조**

```
my_ml_project/
│
├── MLproject
├── conda.yaml
└── train.py
```

**MLproject**: 이 파일은 프로젝트의 구성을 정의합니다. 여기에는 프로젝트의 이름, 환경 설정, 엔트리 포인트 및 파라미터가 포함됩니다.

```yaml
name: My_ML_Project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
    command: "python train.py {alpha}"
```

**conda.yaml**: 이 파일은 프로젝트 실행에 필요한 모든 의존성을 정의합니다.

```yaml
name: my_mlflow_env
channels:
  - defaults
dependencies:
  - python=3.8
  - scikit-learn
  - mlflow
  - numpy
  - pip
```

**train.py**: 이것은 프로젝트의 주 실행 파일입니다. 이 예제에서는 선형 회귀 모델을 훈련합니다.

```py
import mlflow
import mlflow.sklearn
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # 파라미터 받기
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

    # 데이터 준비
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # MLflow 실험 시작
    with mlflow.start_run():
        # 모델 생성 및 훈련
        model = LinearRegression()
        model.fit(X, y)

        # 예측 및 메트릭 계산
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)

        # 파라미터, 메트릭, 모델 로깅
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
```

**프로젝트 실행**

이 프로젝트를 실행하려면, 터미널에서 프로젝트 디렉토리 (my_ml_project)로 이동한
후 다음 명령을 실행합니다:

```
mlflow run . -P alpha=0.3
```

이 명령은 train.py 스크립트를 alpha 파라미터 값 0.3으로 실행합니다. MLflow는
자동으로 필요한 환경을 설정하고 프로젝트를 실행합니다.

## MLflow Models Example

MLflow Models는 머신러닝 모델을 포맷화하고 저장하는 기능을 제공합니다. 이를 통해
모델을 여러 환경에서 쉽게 배포하고 사용할 수 있습니다. MLflow Models는 다양한
머신러닝 프레임워크와 호환되며, 모델을 MLmodel 형식으로 저장합니다.

이 예제에서는 MLflow를 사용하여 머신러닝 모델을 저장하고, MLflow Models 형식으로
모델을 로드하는 방법을 보여줍니다.

**환경 설정**

먼저, 필요한 라이브러리를 설치합니다. Python 환경에서는 다음 명령을 사용하여 설치할 수 있습니다:

```bash
pip install mlflow sklearn
```

**모델 저장 예제**

이 예제에서는 scikit-learn을 사용하여 간단한 선형 회귀 모델을 훈련하고, MLflow로 모델을 저장합니다.

```py
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression

# 데이터 준비
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# 모델 생성 및 훈련
model = LinearRegression()
model.fit(X, y)

# MLflow로 모델 저장
mlflow.sklearn.log_model(model, "my_linear_regression_model")
```

이 코드는 `my_linear_regression_model` 이름으로 MLflow 모델 형식으로 선형 회귀
모델을 저장합니다. 저장된 모델은 MLflow의 추적 서버나 지정된 경로에 저장됩니다.

**모델 로드 및 사용**

저장된 모델을 로드하려면 `mlflow.pyfunc.load_model` 함수를 사용합니다. 모델을
로드한 후, 다른 Python 스크립트나 서비스에서 해당 모델을 사용할 수 있습니다.

```py
import mlflow.pyfunc

# 모델 로드
model_uri = "runs:/<RUN_ID>/my_linear_regression_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# 모델 사용
test_data = np.array([[3, 5]])
prediction = loaded_model.predict(test_data)
print("Prediction:", prediction)
```

`<RUN_ID>`는 모델이 저장된 MLflow 실행(run)의 ID입니다. 이 ID는 MLflow UI나
API를 통해 얻을 수 있습니다.

**주의 사항**

- MLflow 모델은 다양한 소스에서 로드할 수 있습니다. 이 예제에서는 MLflow
  실행에서 모델을 로드하는 방법을 보여줍니다. 모델을 파일 시스템, S3, Azure Blob
  Storage 등 다른 소스에서 로드하는 것도 가능합니다.
- MLflow는 scikit-learn, TensorFlow, PyTorch, Keras 등 다양한 머신러닝
  프레임워크의 모델을 지원합니다. 사용하는 프레임워크에 맞게
  `mlflow.<framework>.log_model` 함수를 사용하여 모델을 저장할 수 있습니다.
- 모델을 로드하려면, 모델이 저장된 위치와 모델 이름 또는 경로를 정확하게
  지정해야 합니다.

## MLflow Model Registry Example

MLflow Model Registry는 MLflow의 중요한 구성 요소로, 모델의 전체 라이프사이클을
관리하는 데 도움을 줍니다. 모델 레지스트리를 사용하면 모델의 등록, 버전 관리,
스테이지 관리 (예: 개발, 스테이징, 프로덕션), 그리고 모델의 협업 및 거버넌스를
할 수 있습니다.

이 예제에서는 간단한 모델을 MLflow Model Registry에 등록하고, 다양한 스테이지로
이동하는 과정을 보여줍니다.

**환경 설정**

먼저, 필요한 라이브러리를 설치합니다. Python 환경에서는 다음 명령을 사용하여 설치할 수 있습니다:

```bash
pip install mlflow sklearn
```

**모델 훈련 및 로깅**

먼저, scikit-learn을 사용하여 간단한 선형 회귀 모델을 훈련하고 MLflow에 로깅합니다.

```py
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression

# 데이터 준비
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# MLflow 실험 시작
with mlflow.start_run():
    # 모델 생성 및 훈련
    model = LinearRegression()
    model.fit(X, y)

    # MLflow로 모델 로깅
    mlflow.sklearn.log_model(model, "my_linear_regression_model")
```

**모델 등록**

MLflow UI를 사용하거나 MLflow API를 사용하여 로깅된 모델을 MLflow Model
Registry에 등록할 수 있습니다. 여기서는 API를 사용한 예를 보여줍니다.

```py
# MLflow에 로깅된 모델의 run_id 및 모델 경로 확인
run_id = mlflow.active_run().info.run_id
model_uri = f"runs:/{run_id}/my_linear_regression_model"

# 모델 레지스트리에 모델 등록
model_name = "MyLinearRegressionModel"
model_version = mlflow.register_model(model_uri, model_name)
```

**모델 스테이지 변경**

등록된 모델의 버전을 다양한 스테이지로 이동할 수 있습니다. 예를 들어, 모델을
'Staging'으로 이동한 후 'Production'으로 이동할 수 있습니다.

```py
# 모델 버전을 Staging으로 이동
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# 모델 버전을 Production으로 이동
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
```

**모델 사용**

레지스트리에서 모델을 로드하여 사용할 수 있습니다.

```py
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# 모델 사용
test_data = np.array([[3, 5]])
prediction = model.predict(test_data)
print("Prediction:", prediction)
```

**주의 사항**

- MLflow Model Registry는 MLflow 서버 또는 Databricks와 같은 지원되는 백엔드를
  사용하여 설정해야 합니다.
- 모델 레지스트리의 사용은 모델 거버넌스 및 협업을 위한 강력한 도구이며, 팀이나
  조직에서 중요한 모델을 관리하는 데 유용합니다.
- 모델을 레지스트리에 등록하고 스테이지를 변경하는 과정은 조직의 워크플로우 및
  거버넌스 정책에 따라 다를 수 있습니다.
