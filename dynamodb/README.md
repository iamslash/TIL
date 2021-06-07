- [Abstract](#abstract)
- [Materials](#materials)
- [Install with docker](#install-with-docker)
- [Basic](#basic)
  - [Features](#features)
    - [Table Structure](#table-structure)
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
- [DynamoDB Data Modeling & Best Practices](#dynamodb-data-modeling--best-practices)
  - [Design Patterns](#design-patterns)
    - [One-to-one](#one-to-one)
    - [One-to-Many](#one-to-many)
    - [Many-to-Many](#many-to-many)
    - [Hierarchical Data Structures](#hierarchical-data-structures)
      - [Table items](#table-items)
      - [JSON Documents](#json-documents)
  - [Multi-value Sorts and Filters](#multi-value-sorts-and-filters)
  - [DynamoDB Limits](#dynamodb-limits)
    - [Capacity and Throught Limits](#capacity-and-throught-limits)
    - [Index and Attribute Limits](#index-and-attribute-limits)
    - [API Limits](#api-limits)
  - [Error Handling in DynamoDB](#error-handling-in-dynamodb)
  - [Ways to Lower DynamoDB Costs](#ways-to-lower-dynamodb-costs)

-------

# Abstract

dynamoDB 는 10 밀리초 미만의 성능을 제공하는 key-value document DB 이다. 하루에 10 조개 이상의 요청을 처리할 수 있고, 초당 2,000 만개 이상의 피크 요청을 지원한다.

Transaction 을 지원한다???

# Materials

* [What is DynamoDB?](https://www.dynamodbguide.com/what-is-dynamo-db/)
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

### Table Structure

* [효과적인 NoSQL (Elasticahe / DynamoDB) 디자인 및 활용 방안 (최유정 & 최홍식, AWS 솔루션즈 아키텍트) :: AWS DevDay2018](https://www.slideshare.net/awskorea/nosql-elasticahe-dynamodb-aws-aws-devday2018)

-----

![](table_structure.jpg)

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

DynamoDB provides 2 kinds of provisioned throughputs. RCU, WCU 는 비용과 관련이 있다.

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

# DynamoDB Data Modeling & Best Practices

## Design Patterns

One-to-One, One-to-Many, Many-to-Many, Hierarchical Data Structures 의 경우에 대해 생각해 보자.

| Relationship | description |
|---|---|
| One-to-One | Simple keys on both entities |
| One-to-Many | Simple Keys on one entity and composite key on the other |
| Many-to-Many | Composite keys or indexes on both entities |

### One-to-one

예를 들어 다음과 같은 Users 테이블이 있다.

| user_id | name | email | SSN |
|---|---|---|---|
| A | John | john@abc.com | 123 |
| B | Mary | mary@abc.com | 124 |
| C | Bill | bill@abc.com | 125 |

예를 들어 다음과 같이 Students, Grades Table 이 있다. Students.student_id, Grades.student_id 는 Primary Key 이다. 

| student_id | name | email | SSN |
|---|---|---|---|
| 1001 | John | john@abc.com | 123 |
| 1002 | Mary | mary@abc.com | 124 |
| 1003 | Bill | bill@abc.com | 125 |

| student_id | grade |
|---|---|
| 1001 | A |
| 1002 | A+ |
| 1003 | A |

예를 들어 다음과 같이 Students, Grades Table 이 있다. Students.student_id, Grades.dept_id 는 Primary Key 이다. Grades.student_id 는 GSI (global Secondary Index) 이다.

| student_id | name | email | SSN |
|---|---|---|---|
| 1001 | John | john@abc.com | 123 |
| 1002 | Mary | mary@abc.com | 124 |
| 1003 | Bill | bill@abc.com | 125 |

| dept_id | student_id | grade |
|---|---|---|
| D1 | 1001 | A |
| D2 | 1002 | A+ |
| D3 | 1003 | A |

### One-to-Many

예를 들어 다음과 같이 Students, Subjects Table 이 있다. Students.student_id 는 Primary key 이다. Subjets.student_id, Subjets.subject 는 composite Primary Key 이다.

| student_id | name | email | SSN |
|---|---|---|---|
| 1001 | John | john@abc.com | 123 |
| 1002 | Mary | mary@abc.com | 124 |
| 1003 | Bill | bill@abc.com | 125 |

| student_id | subject |
|---|---|
| 1001 | Math |
| 1002 | Physics |
| 1003 | Economics |

다음과 같이 Subjects table 를 refactoring 한다. Subjects.subject 는 Primary Partition Key 이다. Subjects.student_id 는 Primary Sort Key 이다. Subjects.student_id 는 GSI Partition Key 이다.

| subject_id | student_id | subject |
|---|---|---|
| S001 | 1001 | Math |
| S002 | 1001 | Physics |
| S003 | 1003 | Economics |

또한 다음과 같이 Subjects table 를 Set 을 사용하여 refactoring 할 수 있다.

| student_id | name | email | SSN | subjects |
|---|---|---|---|---|
| 1001 | John | john@abc.com | 123 | {Math, Physics} |
| 1002 | Mary | mary@abc.com | 124 | {Economics, Civics} |
| 1003 | Bill | bill@abc.com | 125 | {Computer Science, Math} |

다음과 같이 Sort Keys/Composite Keys 혹은 Set Types 를 언제 사용하면 좋은지 정리해 보자.

| Sort Keys/Composite Keys | Set Types |
|---|---|
| Larget item sizes | Small item sizes |
| If querying multiple items whthin a partition key is required | If querying individual item attributes in Sets is NOT needed |

### Many-to-Many

예를 들어 다음과 같이 Students Table 을 살펴보자. student_id 와 subject_id 는 many-to-many 이다. Students.student_id 는 Primary Partition Key 로 한다. Students.subject_id 는 Primary Sort Key 로 한다. Students.student_id 는 GSI Sort Key 로 한다. Students.subject_id 는 GSI Partition Key 로 한다.

| student_id | subject_id | subject |
|---|---|---|
| 1001 | S001 | Math | 
| 1001 | S002 | Physics | 
| 1003 | S003 | Economics | 
| 1003 | S001 | Math | 

### Hierarchical Data Structures

Table items 과 JSON Documents 를 이용하여 표현할 수 있다.

#### Table items

예를 들어 다음과 같이 Curriculum table 이 있다. Curriculum.curriculum_id 는 Partition Key 이다. Curriculum.type 은 Sort Key 이다.

| curriculum_id | type | attributes_1 | ... | attributes_n |
|----|---|---|---|---|
| Medical | Radiology | ... | ... | ... |
| Medical | Dentistry | ... | ... | ... |
| Medical | ... | ... | ... | ... |
| Engineering | Computer Science | ... | ... | ... |
| Engineering | Electronics | ... | ... | ... |
| Engineering | Mechanical Engineering | ... | ... | ... |
| Engineering | ... | ... | ... | ... |
| Journalism | Newspaper Journalism | ... | ... | ... |
| Journalism | Investigative Journalism | ... | ... | ... |
| Journalism | Sports Journalism | ... | ... | ... |
| Journalism | ... | ... | ... | ... |

#### JSON Documents

예를 들어 다음과 같이 Products table 이 있다. Products.product_id 는 Partition Key 이다. metadata 는 key value 의 모음을 JSON 으로 저장하고 있다.

| product_id | metadata |
| P001 | `{type: "Electronics", model: "PQR", weith: "1.05"}` |
| P002 | `{publisher: "ABC", type: "Electronics", model: "PQR", weith: "1.05"}` |

## Multi-value Sorts and Filters

Sort Key 를 이용하면 하나의 Partition 안에서 Sort Key 로 rows 를 정렬할 수 있다. 

만약 여러 column 을 기준으로 정렬해야 하는 경우를 살펴보자. 다음과 같은 Address table 이 있다. country, state, city 를 기준으로 정렬을 해보자.

| customer | addr_id | country | state | city | zip |street | 
|--|--|--|--|--|--|--|
| John | 1 | US | CA | San Francisco | ... | ... | ... |
| Tom | 1 | US | CA | San Diego | ... | ... | ... |
| John | 2 | US | CA | San Diego | ... | ... | ... |
| Sara | 1 | US | FL | Miami | ... | ... | ... |

첫번 째 방법은 다음과 같이 GSI 3 개를 설정하는 것이다.

* GSI 1
  * Address.customer: GSI 1 Partition Key
  * Address.country: GSI 1 Sort Key
* GSI 2
  * Address.customer: GSI 2 Partition Key
  * Address.state: GSI 2 Sort Key
* GSI 3
  * Address.customer: GSI 3 Partition Key
  * Address.city: GSI 3 Sort Key

다음과 같이 Query 한다. 이렇게 여러개의 Sort Key 를 사용하려 filtering 하는 것을 Multi-valoue Filters 라고 한다.

```
Filter On:
state = "CA" and city = "San Diego"
```

한편 다음과 같이 sort key 를 design 하면 성능을 개선할 수 있다.

| customer | addr_id | country_state_city | zip |street | 
|--|--|--|--|--|--|
| John | 1 | US `|` CA `|` San Francisco | ... | ... |
| Tom | 1 | US `|` CA `|` San Diego | ... | ... | 
| John | 2 | US `|` CA `|` San Diego | ... | ... |
| Sara | 1 | US `|` FL `|` Miami | ... | ... | 

다음과 같이 Query 한다. 이렇게 하나의 column 에 multi-value 를 사용하고 Sort key 로 설정한 것을 Multi-value Sorts 라고 한다.

```
Query On:
customer = "John" and country_state_city BEGINS_WITH "US | CA |"
```

## DynamoDB Limits

* [Service, Account, and Table Quotas in Amazon DynamoDB @ aws](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Limits.html)

###  Capacity and Throught Limits

* 4KB per RCU
* 1KB per WCU
* 10GB per partition
* 3000 RCUs or 1000 WCUs per partition
* Minimum 1 RCU and 1 WCU per table or index

* Max 40,000 RCUs and WCUs each per table (US East region)
* Max 10,000 RCUs and WCUs each per table (Other regions)
* Max 80,000 RCUs and WCUs each per account (US East region)
* Max 20,000 RCUs and WCUs each per account (Other regions)
* Max limits are soft limits and can be increase on request

* No limit on scaling up table capacity
* Max scale downs limited to 4 times per calendar day (UTC timezone)
* Additional 1 scale down if no scale downs in last 4 hours
* Effectively 9 scale downs per day
* Max 256 tables per region (soft limit)

### Index and Attribute Limits

* 5 local secondary indexes per table
* 5 global secondary indexes per table
* Max 20 user-specified projected attributes across all secondary indexes of the table
* Max size of partition key = 2 KB
* Max size of sort key = 1 KB
* Max size of all items per partition key = 10 GB (including all LSIs)
* Max size of a table item = 400 KB
* For nested attributes, max possible nesting is 32 levels deep

### API Limits

* Max 10 simultaneous requests for table-level operations (CreateTable, UpdateTable and DeleteTable)
* Max 100 items (up to 16 MB in size) returned per **BatchGetItem** request
* Max 25 PutItem or DeleteItem requests (up to 16 MB in size) per **BatchWriteItem** request
* Max 1 MB data returned per query or scan request

## Error Handling in DynamoDB

* HTTP 400
  * Error in our request
  * Error Authentication failure
  * Missing required parameters
* HTTP 500
  * 500 Server Side Error
  * 503 Service not available
* Exceptions
  * Access Denied Exception
  * Conditional Check Failed Exception
  * Item Collection Size Limit Exceeded Exception
  * Limit Exceeded Exception
  * Resource In use Exception
  * Validation Exception
  * Provisioned Throughput Exceeded Exception
    * Error Retries
    * Exponential Backoff
  
## Ways to Lower DynamoDB Costs
