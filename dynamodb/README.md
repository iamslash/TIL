- [Materials](#materials)
- [Install with docker](#install-with-docker)
- [Basic](#basic)
  - [Features](#features)
    - [Data Types](#data-types)
    - [Table Index](#table-index)
    - [Secondary Index](#secondary-index)
    - [Consistency](#consistency)
    - [Provisioned Throughput](#provisioned-throughput)
    - [Read Data](#read-data)
  - [AWS CLI commands](#aws-cli-commands)
  - [Examples of Schema Design](#examples-of-schema-design)
    - [Fruits](#fruits)
      - [Fruits](#fruits-1)
      - [Query](#query)
    - [Weekly Rank](#weekly-rank)
      - [UsersLeaderboard](#usersleaderboard)
      - [FriendsLeaderboard](#friendsleaderboard)
      - [Query](#query-1)

-------

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

* **Scala Data Types** : Number, String, Binary.
* **Multi Valued Types** : Number Set, String Set, Binary Set.

### Table Index

Table Index 는 Primary Key 와 같다. Primary Key 는 Hash key, Hash-Range key 두가지가 있다. 

* **Hash Key** : Attribute 하나를 Key 로 사용한다. Scala Data Type 만 가능하다. Multi Valued Type 은 불가능하다.
* **Hash-Range Key** : Attribute 두개를 Key 로 사용한다. 첫번째 속성은 Hash Key 로 사용하고 두번째 속성은 Range Key 로 사용한다. Hash Key 와 Range Key 를 합한 것이다.

### Secondary Index

Primary Key 이외의 Index 를 Secondary Index 라고 한다. Primary Key 만으로는 검색기능이 부족하다. Secondary Index 는 사용이 빈번하기 때문에 성능을 위해 읽기/쓰기 용량 유닛을 따로 설정할 수 있다. Local Secondary Index 와 Global Secondary Index 가 있다.

* **Local Secondary Index** : 하나의 Partition 에서만 유효한 Index 이다. Hash key 는 Table Index 의 Hash Key 와 같다. Range key 는 다르게 설정한다. Table 당 5 개 까지 가능하다. Table 이 생성될 때 생성해야 한다. Table 이 생성된 후 추가, 수정, 삭제가 불가능하다. Table 의 Index 설정화면에서 Hash and Range Key 를 선택할 때만 생성할 수 있다.
* **Global Secondary Index** : 여러개의 Partition 에 걸쳐서 유효한 Index 이다. Hash Key, Range Key 모두 Table Index 와 다르게 설정한 것이다. Range Key 는 생략가능하다. Table 당 5 개 까지 가능하다. Table 이 생성될 때 생성해야 한다. Table 이 생성된 후 추가, 수정, 삭제가 불가능하다. 

### Consistency

DynamoDB provides 2 kinds of consistency.

* **Strongly Consistent Read** : 최근 완료된 쓰기결과가 모두 반영된 데이터를 읽는다.
* **Eventually Consistent Read** : 최근 완료된 쓰기결과가 반영되지 못했을 수 있다. 쓰기가 데이터의 모든 복사본에 반영되는 것은 1 초 내에 이루어진다. 최신 데이터를 읽으려면 짧은 시간 내에 읽기를 반복해야 한다.

### Provisioned Throughput

DynamoDB provides 2 kinds of provisioned throughputs.

* Read Capacity Units: 초당 1KB 단위로 읽을 수 있는 능력 (Eventually Consistent Read 는 Stronly Consistent Read 보다 2 배이다.)
* Write Capacity Units: 초당 1KB 단위로 쓸 수 있는 능력

예) 512 바이트 (1KB 로 반올림) 를 초당 200 개 항목을 읽으면(쓰면), 1KB x 200 = 200 유닛
예) 1.5 KB (2KB로 반올림 됨) 를 초당 200 개 항목을 읽으면(쓰면), 2KB x 200 = 400 유닛
예) Strongly Consistent Read 는 1000 읽기 용량 유닛으로 1KB 짜리 아이템을 초당 1000 번 읽을 수 있으며 Eventually Consistent Read 는 500 읽기 용량 유닛으로 1KB 짜리 아이템을 1000 번 읽을 수 있습니다.

### Read Data

dynamoDB provides 2 ways to read data and they limit the result as 1 MB.

* Scan: Gather all data without condition.
* Query: Gather data with Hash Key, Range Key conditions. Range Key can be removed.

## AWS CLI commands

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

## Examples of Schema Design

### Fruits

* [AWS CLI로 DynamoDB 다루기](https://www.daleseo.com/aws-cli-dynamodb/)

----

#### Fruits

| Name  | Data-type | Table Index | Local Secondary Index | Global Secondary Index |
| ----- | --------- | ----------- | --------------------- | ---------------------- |
| Id    | Number    | Hash Key    |                       |                        |
| Name  | String    |             |                       |                        |
| Price | Number    |             |                       |                        |

#### Query

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

### Weekly Rank

#### UsersLeaderboard

| Name     | Data-type | Table Index | Local Secondary Index | Global Secondary Index |
| -------- | --------- | ----------- | --------------------- | ---------------------- |
| Id       | Number    | Hash Key    | Hash Key              |                        |
| Name     | String    |             |                       |                        |
| TopScore | Number    |             |                       | Range Key              |
| Week     | String    | Range Key   | Range Key             | Hash Key               |
	 	 	 	 
#### FriendsLeaderboard	 

| Name            | Data-type | Table Index | Local Secondary Index | Global Secondary Index |
| --------------- | --------- | ----------- | --------------------- | ---------------------- |
| Id              | Number    | Hash Key    |                       |                        |
| Name            | String    |             |                       |                        |
| Score           | Number    |             |                       | Range Key              |
| FriendIdAndWeek | String    | Range Key   |                       | Hash Key               |
	 	 	 	 
#### Query

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
