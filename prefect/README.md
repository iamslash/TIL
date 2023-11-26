# Abstract

Prefect는 Airflow와 유사한 작업 오케스트레이션 및 데이터 파이프라인 자동화를
위한 소프트웨어 도구입니다. 이 두 도구는 데이터 엔지니어링과 작업 스케줄링에서
중요한 역할을 합니다. 그러나 Prefect와 Airflow는 설계 철학과 기능적 측면에서
차이가 있습니다.

# Basic

## Features

- 사용자 친화적: Prefect는 사용하기 쉽게 설계되었으며, 특히 Python 사용자에게
  친숙합니다. 파이프라인을 정의하는 데 Python 코드를 사용합니다.
- 오류 처리: Prefect는 오류 처리와 재시도 로직에 중점을 둡니다. 이는 데이터
  파이프라인에서 예상치 못한 오류를 처리하는 데 유용합니다.
- 클라우드 지원: Prefect는 클라우드 기반의 오케스트레이션을 제공하여 사용자가
  자체 인프라를 관리할 필요가 없도록 합니다.
- 인터페이스와 통합: 사용자 친화적인 인터페이스를 제공하며, 다양한 데이터 소스와 통합이 용이합니다.

**Airflow와의 차이점:**

- 설계 철학: Airflow는 주로 스케줄링과 작업 종속성 관리에 중점을 둡니다. 반면,
  Prefect는 오류 처리와 데이터 파이프라인의 간소화에 더 많은 초점을 맞춥니다.
- 구현의 복잡성: Airflow는 강력하지만, 초기 설정과 관리가 복잡할 수 있습니다.
  Prefect는 사용의 용이성을 목표로 하여, 보다 간단하고 직관적인 설정을
  제공합니다.
- 커뮤니티와 지원: Airflow는 오랜 기간 동안 널리 사용되어 왔기 때문에 큰
  커뮤니티와 광범위한 지원을 자랑합니다. Prefect는 비교적 새로운 도구이지만,
  사용자 친화적인 접근 방식으로 빠르게 성장하고 있습니다.

## Prefect Server

Prefect는 사용자 인터페이스(UI)를 통해 플로우의 실행 상태와 세부 정보를
모니터링할 수 있는 기능을 제공합니다. 이 UI를 사용하려면 먼저 Prefect 서버와
UI를 로컬에 설치하고 실행해야 합니다. 다음은 Prefect UI를 설정하고 사용하는
방법에 대한 단계별 지침입니다:

**Prefect 서버와 UI 설치 및 실행:**

서버와 UI를 실행하려면, 터미널에서 다음 명령어를 실행합니다:

```bash
prefect server start
```

**웹 브라우저에서 Prefect UI 열기:**

Prefect 서버가 성공적으로 시작되면, 기본적으로 http://localhost:8080 주소에서
UI에 접근할 수 있습니다. 웹 브라우저를 열고 이 주소를 입력하면 Prefect의
대시보드에 접근할 수 있습니다. 

**Prefect UI 사용:**

- 플로우 모니터링: UI에서는 실행 중이거나 완료된 플로우의 상태를 볼 수 있습니다.
  각 플로우의 실행 세부 사항, 로그, 실행 결과 등을 확인할 수 있습니다.
- 플로우 등록 및 스케줄링: 새로운 플로우를 등록하고 스케줄을 설정할 수 있습니다.
  이를 통해 주기적으로 플로우를 실행하도록 스케줄링 할 수 있습니다.
- 시스템 상태 확인: Prefect 서버의 건강 상태와 활성 에이전트 등의 정보도 확인할
  수 있습니다.

Prefect UI는 데이터 파이프라인의 상태를 시각적으로 관리하고 모니터링하는 데
유용한 도구입니다. 그러나 모든 기능을 사용하려면 Prefect Cloud와 같은 추가적인
서비스를 구성할 수도 있습니다.

## Simple ETL Example

이 예제에서는 Prefect를 사용하여 기본적인 데이터 처리 작업을 스케줄링하고 실행하는 방법을 보여줍니다.

**Prefect 설치:**

먼저 Prefect를 설치해야 합니다. 이는 Python의 pip를 통해 쉽게 설치할 수 있습니다.

```bash
pip install prefect
```

**Prefect 플로우 생성:**

이 예제에서는 간단한 데이터 처리 작업을 정의하고 Prefect 플로우로 구성합니다.

```py
from prefect import task, Flow

@task
def extract():
    # 데이터 추출 로직
    data = {"data": [1, 2, 3, 4, 5]}
    return data

@task
def transform(data):
    # 데이터 변환 로직
    transformed_data = [x * 2 for x in data["data"]]
    return transformed_data

@task
def load(transformed_data):
    # 데이터 로드 로직
    print("Final data:", transformed_data)

with Flow("ETL-pipeline") as flow:
    e = extract()
    t = transform(e)
    l = load(t)

# 플로우 실행
flow.run()
```

이 코드는 간단한 ETL (Extract, Transform, Load) 프로세스를 구현합니다. extract
함수는 데이터를 추출하고, transform 함수는 데이터를 변환하며, load 함수는 변환된
데이터를 로드합니다. 이러한 각 단계는 Prefect의 @task 데코레이터로 정의된
태스크입니다. 이 태스크들은 Flow 내에서 연결되어 전체 파이프라인을 형성합니다.

**플로우 실행:**

마지막 줄의 `flow.run()`은 이 플로우를 실행합니다. 이는 로컬 환경에서 실행되며,
각 태스크의 실행 순서와 결과를 관리합니다.

이 예제는 Prefect의 기본적인 사용 방법을 보여줍니다. 실제 사용 시에는 데이터
소스, 변환 로직, 데이터 저장 방법 등에 따라 태스크의 구현이 달라질 수 있습니다.
또한, Prefect는 클라우드 기반 실행, 상태 모니터링, 오류 처리 등의 고급 기능을
제공합니다.
