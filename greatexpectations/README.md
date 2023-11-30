# Abstract

Great Expectations은 데이터 파이프라인에서 데이터 품질을 관리하는 데 도움이 되는
오픈 소스 Python 라이브러리입니다. 기본적으로 데이터 프레임에 대한 패턴과 규칙을
정의하여 여러 데이터셋에 대해 스스로 검증할 수 있는 방법을 제공합니다. 이를
사용하면 데이터 품질을 보장하고 문제를 빨리 발견할 수 있습니다. 코드, 설계 및
시스템에 대한 실수가 데이터 파이프라인에 영향을 미치는 것을 방지하는 데 도움이
됩니다.

# Basic

## Great Expectations Simple Example

```py
# 1. 라이브러리를 설치하고 불러온다
!pip install great_expectations
import great_expectations as ge

# 2. Pandas DataFrame 데이터 생성
import pandas as pd
data = {'name': ['Alice', 'Bob', 'Carol', 'David'],
        'age': [25, 34, 29, 41],
        'city': ['New York', 'San Francisco', 'Los Angeles', 'Seattle']}
df = pd.DataFrame(data)

# 3. DataFrame -> Great Expectations DataFrame 캐스팅
gedf = ge.from_pandas(df)

# 4. 데이터셋에 기대하는 패턴이나 규칙을 설정
gedf.expect_column_values_to_be_unique('name')
gedf.expect_column_values_to_not_be_null('age')
gedf.expect_column_values_to_be_of_type('city', str)
gedf.expect_column_values_to_be_in_set('city', ['New York', 'San Francisco', 'Los Angeles', 'Seattle'])

# 5. 기대를 정의 및 확인
result = gedf.validate()
print(result)

# 이 명령이 실행되면 기대할 수 있는 패턴 및 규칙에 따라 검증 결과를 얻게 됩니다.
# 모든 기대 사항이 충족되면 결과는 success=True로 표시되고, 그렇지 않으면 success=False로 표시되어 프로세스 진행시 요구 사항이 충족되지 않음을 알 수 있습니다.
```

이 코드는 다음과 같은 절차를 거칩니다:

- Great Expectations 라이브러리를 설치하고 불러옵니다.
- Pandas DataFrame 데이터를 생성합니다.
- 생성된 DataFrame에 Great Expectations를 적용합니다.
- 데이터셋에 담긴 값들이 충족해야하는 기대 사항을 설정합니다.
- 기대 사항들을 검증합니다.

이렇게 간단한 예제 코드를 통해 Great Expectations의 기본 사용 방법 및 검증
메서드의 작동을 확인할 수 있습니다. 이렇게 데이터의 질을 관리하면 후속
프로세스에서 오류를 효율적으로 찾아낼 수 있습니다.
