# Materials

* [DynamoDB의 데이터 모델 @ pyrasis](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter14/01)
  * [DynamoDB에 맞는 데이터 구조 설계하기 @ pyrasis](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter14/02)
  * [DynamoDB 테이블 생성하기 @ pyrasis](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter14/03)
* [dynamoDB CLI @ AWS](https://docs.aws.amazon.com/cli/latest/reference/dynamodb/index.html)

# Install with docker

* [amazon/dynamodb-local](https://hub.docker.com/r/amazon/dynamodb-local/)

```console
$ docker pull amazon/dynamodb-local
$ docker run -d -p 8000:8000 --rm --name my-dynamodb amazon/dynamodb-local
```

# Basic

## Features

### Data Types

* [Supported Data Types @ AWS](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBMapper.DataTypes.html)

----

Scala Data Types are Number, String, Binary.
Multi Valued Types are Number Set, String Set, Binary Set.

### Index

Primary Key 로 Index 를 생성한다. Primary Key 는 Hash key, Range key 두가지가 있다. 

* **Hash Key** : 속성하나를 기본 키로 사용한다. Primary Key 의 값은 Scala Data 만 가능하다. Multi Value 는 불가능하다.
* **Hash, Range Key** : 속성 두개를 기본 키로 사용한다. 첫번째 속성은 기본 키로 사용하고 두번째 속성은 범위 기본키로 사용하여 복합적으로 사용한다.

### Secondary Index

Primary Key 이외의 index 를 Secondary Index 라고 한다. Primary Key 만으로는 검색이능이 부족하다. Secondary Index 는 사용이 빈번하기 때문에 성능을 위해 읽기/쓰기 용량 유닛을 따로 설정할 수 있다.

* **Local Secondary Index** : hash key 는 Table index 의 hash key 와 같다. range key 는 다르게 설정한다. Table 당 5 개 까지 가능하다. Table 이 생성될 때 생성해야 한다. Table 이 생성된 후 추가, 수정, 삭제가 불가능하다. Table 에서 hash key, range key 를 사용할 때만 생성할 수 있다.
* **Global Secondary Index** :  hash key, range key 모두 Table 의 index 와 다르게 설정한 것이다. range key 는 생략가능하다. Table 당 5 개 까지 가능하다. Table 이 생성될 때 생성해야 한다. Table 이 생성된 후 추가, 수정, 삭제가 불가능하다. 

### Consistency

DynamoDB provides 2 kinds of consistency.

* **Strongly Consistent Read** : 최근 완료된 쓰기결과가 모두 반영된 데이터를 읽는다.
* **Eventually Consistent Read** : 최근 완료된 쓰기결과가 반영되지 못했을 수 있다. 쓰기가 데이터의 모든 복사본에 반영되는 것은 1 초 내에 이루어진다. 최신 데이터를 읽으려면 짧은 시간 내에 읽기를 반복해야 한다.

### Provisioned Throughput

DynamoDB provides 2 kinds of provisioned throughputs.

* Read Capacity Units: 초당 읽은 아이템 수 x KB 단위 아이템 크기(반올림) (Eventually Consistent Read 를 사용하는 경우 초당 읽은 아이템 용량은 두 배가됩니다.)
* Write Capacity Units: 초당 쓴 아이템 수 x KB 단위 아이템 크기(반올림)

예) 512 바이트 (1KB 로 반올림) 를 초당 200 개 항목을 읽으면(쓰면), 1KB x 200 = 200 유닛
예) 1.5 KB (2KB로 반올림 됨) 를 초당 200 개 항목을 읽으면(쓰면), 2KB x 200 = 400 유닛

예) Strongly Consistent Read 는 1000 읽기 용량 유닛으로 1KB 짜리 아이템을 초당 1000 번 읽을 수 있으며 Eventually Consistent Read 는 500 읽기 용량 유닛으로 1KB 짜리 아이템을 1000 번 읽을 수 있습니다.

### Read Data

dynamoDB provides 2 ways to read data and they limit the result as 1 MB.

* Scan: Gather all data without condition.
* Query: Gather data with hash type key, range type key conditions. range type key can be removed.

# AWS CLI commands

* [Amazon DynamoDB 로컬 환경에서 사용하기 (feat. Docker)](https://medium.com/@byeonggukgong/using-amazon-dynamodb-in-local-environment-feat-docker-fafbb420e161)

```console
$ aws dynamodb list-tables --endpoint-url http://localhost:8000

$ aws dynamodb create-table \
    --table-name <table-name> \
    --attribute-definitions \
        AttributeName=<attr-name>,AttributeType=<attr-type> \
        ... \
    --key-schema \
        AttributeName=<attr-name>,KeyType=<key-type> \
        ... \
    --provisioned-throughput \
        ReadCapacityUnits=1,WriteCapacityUnits=1 \
    --endpoint-url http://localhost:8000

$ aws dynamodb describe-table
    --table-name <table-name> \
    --endpoint-url http://localhost:8000   

$ aws dynamodb put-item
    --table-name <table-name> \
    --item \
       '{ \
            "<attr-name>": {"<attr-type>": <content>}, \
            ... \
        }' \
    --endpoint-url http://localhost:8000     

$ aws dynamodb get-item \
    --table-name <table-name> \
    --key \
        '{ \
            "<attr-name>": {"<attr-type>": <content>}, \
            ... \
        }' \
    --endpoint-url http://localhost:8000   

$ aws dynamodb delete-item \
    --table-name <table-name> \
    --key \
       '{ \
            "<attr-name>": {"<attr-type>": "content"}, \
            ... \
        }'
    --endpoint-url http://localhost:8000    

$ aws dynamodb delete-table \
    --table-name <table-name> \
    --endpoint-url http://localhost:8000    
```

# Examples of Schema Design

## Fruits

* [AWS CLI로 DynamoDB 다루기](https://www.daleseo.com/aws-cli-dynamodb/)

```bash
# Create table
$ aws dynamodb create-table --table-name Fruits --attribute-definitions AttributeName=Id,AttributeType=S --key-schema AttributeName=Id,KeyType=HASH --provisioned-throughput ReadCapacityUnits=1,WriteCapacityUnits=1 --endpoint-url http://localhost:8000
{
    "TableDescription": {
        "AttributeDefinitions": [
            {
                "AttributeName": "Id",
                "AttributeType": "S"
            }
        ],
        "TableName": "Fruits",
        "KeySchema": [
            {
                "AttributeName": "Id",
                "KeyType": "HASH"
            }
        ],
        "TableStatus": "ACTIVE",
        "CreationDateTime": 1595252039.257,
        "ProvisionedThroughput": {
            "LastIncreaseDateTime": 0.0,
            "LastDecreaseDateTime": 0.0,
            "NumberOfDecreasesToday": 0,
            "ReadCapacityUnits": 1,
            "WriteCapacityUnits": 1
        },
        "TableSizeBytes": 0,
        "ItemCount": 0,
        "TableArn": "arn:aws:dynamodb:ddblocal:000000000000:table/Fruits"
    }
}

# Describe table
$ aws dynamodb describe-table --table-name Fruits --endpoint-url http://localhost:8000   
{
    "Table": {
        "AttributeDefinitions": [
            {
                "AttributeName": "Id",
                "AttributeType": "S"
            }
        ],
        "TableName": "Fruits",
        "KeySchema": [
            {
                "AttributeName": "Id",
                "KeyType": "HASH"
            }
        ],
        "TableStatus": "ACTIVE",
        "CreationDateTime": 1595252039.257,
        "ProvisionedThroughput": {
            "LastIncreaseDateTime": 0.0,
            "LastDecreaseDateTime": 0.0,
            "NumberOfDecreasesToday": 0,
            "ReadCapacityUnits": 1,
            "WriteCapacityUnits": 1
        },
        "TableSizeBytes": 0,
        "ItemCount": 0,
        "TableArn": "arn:aws:dynamodb:ddblocal:000000000000:table/Fruits"
    }
}

# Put data
$ aws dynamodb put-item --table-name Fruits --item file://fruits.json --endpoint-url http://localhost:8000 
$ aws dynamodb put-item --table-name Fruits --item '{"Id": {"S": "A1"}, "Name": {"S": "Apple"}, "Price": {"N": "1000"}}' --endpoint-url http://localhost:8000   

# Get data
$ aws dynamodb get-item --table-name Fruits --key '{"Id": {"S": "A1"}}'  --endpoint-url http://localhost:8000   

# Update data
$ aws dynamodb update-item --table-name Fruits \
  --key '{"Id": {"S": "A1"}}' \
  --update-expression 'SET Price=:Price' \
  --expression-attribute-values '{":Price": {"N": "2000"}}' \
  --endpoint-url http://localhost:8000    

# Delete data
$ aws dynamodb delete-item --table-name Fruits \
  --key '{"Id": {"S": "A1"}}' \
  --endpoint-url http://localhost:8000      

# WRite batch data
$ aws dynamodb batch-write-item --request-items file://fruits.json \
  --endpoint-url http://localhost:8000      

# Search data
$ aws dynamodb query --table-name Fruits \
  --key-condition-expression 'Id = :Id' \
  --expression-attribute-values '{":Id": {"S": "B1"}}' \
  --endpoint-url http://localhost:8000          

# Scan data
$ aws dynamodb scan --table-name Fruits \
  --filter-expression 'Price < :Price' \
  --expression-attribute-values '{":Price": {"N": "3000"}}' \
  --return-consumed-capacity TOTAL   \
  --endpoint-url http://localhost:8000          

# Delete table
$ aws dynamodb delete-table --table-name Fruits   \
  --endpoint-url http://localhost:8000          
```

## Weekly Rank

### UsersLeaderboard
	 	 	 	 
| 키 이름(값 형식)   | Id(Number)   | Name(String) | TopScore(Number) | Week(String) |
| ------------------ | ------------ | ------------ | ---------------- | ------------ |
| 테이블 인덱스      | 해시 기본 키 |              |                  | 범위 기본 키 |
| 로컬 보조 인덱스   | 해시 키      |              |                  | 범위 키      |
| 글로벌 보조 인덱스 |              |              | 범위 키          | 해시 키      |

### FriendsLeaderboard	 

| 키 이름(값 형식)   | Id(Number)   | Name(String) | Score(Number) | FriendIdAndWeek(String) |
| ------------------ | ------------ | ------------ | ------------- | ----------------------- |
| 테이블 인덱스      | 해시 기본 키 |              |               | 범위 기본 키            |
| 글로벌 보조 인덱스 |              |              | 범위 키       | 해시 키                 |

### Query

```bash
$ aws dynamodb create-table \
    --table-name UsersLeaderboard \
    --attribute-definitions \
        AttributeName=Id,AttributeType=N \
        AttributeName=Name,AttributeType=S \
        AttributeName=TopScore,AttributeType=N \
        AttributeName=Week,AttributeType=S \
    --key-schema \
        AttributeName=Id,KeyType=HASH \
        AttributeName=Id,KeyType=RANGE \
    --provisioned-throughput \
        ReadCapacityUnits=1,WriteCapacityUnits=1 \
    --endpoint-url http://localhost:8000

$ aws dynamodb describe-table
    --table-name UsersLeaderboard \
    --endpoint-url http://localhost:8000   
```
