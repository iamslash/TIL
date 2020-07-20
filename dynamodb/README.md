# Materials

* [DynamoDB @ pyrasis](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter30/09)
  * [확장 가능한 NoSQL 분산 데이터베이스를 제공하는 DynamoDB @ pyrasis](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter14)
  * [DynamoDB의 데이터 모델](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter14/01)
  * []()
  * []()
  * []()
  * []()
  * []()

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

Primary Key 로 Index 를 생성한다. Primary Key 는 Hash 형식, Range 형식 두가지가 있다.

### Secondary Index


### Consistency

DynamoDB provides 2 kinds of consistency.

* Strongly Consistent Read
* Eventually Consistent Read

### Provisioned Throughput

DynamoDB provides 2 kinds of provisioned throughputs.

* Read Capacity Units: 초당 읽은 아이템 수 x KB 단위 아이템 크기(근사치 반올림) (Eventually Consistent Read를 사용하는 경우 초당 읽은 아이템 용량은 두 배가됩니다.)
* Write Capacity Units: 초당 쓴 아이템 수 x KB 단위 아이템 크기(근사치 반올림)

예) 512 바이트 (1KB 로 반올림) 를 초당 200개 항목을 읽으면(쓰면), 1KB x 200 = 200 유닛 1.5KB(2KB로 반올림 됨)를 초당 200개 항목을 읽으면(쓰면), 2KB x 200 = 400 유닛

예) Strongly Consistent Read 는 1000 읽기 용량 유닛으로 1KB 짜리 아이템을 초당 1000번 읽을 수 있으며 Eventually Consistent Read 는 500 읽기 용량 유닛으로 1KB 짜리 아이템을 1000번 읽을 수 있습니다.

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

## Weekly Rank

```

```


