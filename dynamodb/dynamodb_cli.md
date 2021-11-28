- [Abstract](#abstract)
- [References](#references)
- [Run DynamoDB local](#run-dynamodb-local)
- [Working with Tables](#working-with-tables)
- [Working with Items](#working-with-items)
  - [Basics](#basics)
  - [Bath Operations](#bath-operations)
  - [Atomic Counters](#atomic-counters)
  - [Conditional Writes](#conditional-writes)
  - [Specifying Item Attributes](#specifying-item-attributes)
    - [Nested Attributes](#nested-attributes)
    - [Document Paths](#document-paths)
  - [Projection Expressions](#projection-expressions)
  - [Condition Expressions](#condition-expressions)
  - [Update Expressions](#update-expressions)
    - [SET - Modifying or Adding Item Attributes](#set---modifying-or-adding-item-attributes)
    - [REMOVE - Deleting Attributes from an Item](#remove---deleting-attributes-from-an-item)
    - [ADD - Updating Numbers and Sets](#add---updating-numbers-and-sets)
    - [DELETE - Removing Elements from a Set](#delete---removing-elements-from-a-set)
- [Working with Queries](#working-with-queries)
  - [Key Condition Expressions](#key-condition-expressions)
  - [Filter Expressions for Query](#filter-expressions-for-query)
  - [Paginating Table Query Results](#paginating-table-query-results)
- [Working with Scans](#working-with-scans)
  - [Filter Expressions for Scan](#filter-expressions-for-scan)
  - [Paginating the Results](#paginating-the-results)
  - [Counting the Items in the Results](#counting-the-items-in-the-results)
- [Working with Transactions](#working-with-transactions)
  - [TransactWriteItems API](#transactwriteitems-api)
  - [TransactGetItems API](#transactgetitems-api)
  - [Isolation Levels for DynamoDB Transactions](#isolation-levels-for-dynamodb-transactions)
  - [Transaction Conflict Handling in DynamoDB](#transaction-conflict-handling-in-dynamodb)
  - [Best Practices for Transactions](#best-practices-for-transactions)
  - [Using Transactional APIs with Global Tables](#using-transactional-apis-with-global-tables)

----

# Abstract

DynamoDB commands 를 정리한다.

# References

* [Working with Tables, Items, Queries, Scans, and Indexes @ amazon](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithDynamo.html)

# Run DynamoDB local

```bash
$ mkdir -p ~/my/dynamodb/sample
$ cd ~/my/dynamodb/sample

$ docker run --rm --name my-dynamodb -p 8000:8000 -d amazon/dynamodb-local -jar DynamoDBLocal.jar -sharedDb -dbPath .
```

# Working with Tables

```bash
# Create a Provisioned Table
$ aws dynamodb create-table \
    --endpoint-url http://localhost:8000 \
    --table-name Music \
    --attribute-definitions \
        AttributeName=Artist,AttributeType=S \
        AttributeName=SongTitle,AttributeType=S \
    --key-schema \
        AttributeName=Artist,KeyType=HASH \
        AttributeName=SongTitle,KeyType=RANGE \
    --provisioned-throughput \
        ReadCapacityUnits=10,WriteCapacityUnits=5
{
    "TableDescription": {
        "AttributeDefinitions": [
            {
                "AttributeName": "Artist",
                "AttributeType": "S"
            },
            {
                "AttributeName": "SongTitle",
                "AttributeType": "S"
            }
        ],
        "TableName": "Music",
        "KeySchema": [
            {
                "AttributeName": "Artist",
                "KeyType": "HASH"
            },
            {
                "AttributeName": "SongTitle",
                "KeyType": "RANGE"
            }
        ],
        "TableStatus": "ACTIVE",
        "CreationDateTime": "2021-11-28T16:08:43.236000+09:00",
        "ProvisionedThroughput": {
            "LastIncreaseDateTime": "1970-01-01T09:00:00+09:00",
            "LastDecreaseDateTime": "1970-01-01T09:00:00+09:00",
            "NumberOfDecreasesToday": 0,
            "ReadCapacityUnits": 10,
            "WriteCapacityUnits": 5
        },
        "TableSizeBytes": 0,
        "ItemCount": 0,
        "TableArn": "arn:aws:dynamodb:ddblocal:000000000000:table/Music"
    }
}        

# Create an On-Demand Table
$ aws dynamodb create-table \
    --endpoint-url http://localhost:8000 \
    --table-name Music \
    --attribute-definitions \
        AttributeName=Artist,AttributeType=S \
        AttributeName=SongTitle,AttributeType=S \
    --key-schema \
        AttributeName=Artist,KeyType=HASH \
        AttributeName=SongTitle,KeyType=RANGE \
    --billing-mode=PAY_PER_REQUEST

# Desribe a Table
$ aws dynamodb describe-table --table-name Music \
    --endpoint-url http://localhost:8000 
{
    "Table": {
        "AttributeDefinitions": [
            {
                "AttributeName": "Artist",
                "AttributeType": "S"
            },
            {
                "AttributeName": "SongTitle",
                "AttributeType": "S"
            }
        ],
        "TableName": "Music",
        "KeySchema": [
            {
                "AttributeName": "Artist",
                "KeyType": "HASH"
            },
            {
                "AttributeName": "SongTitle",
                "KeyType": "RANGE"
            }
        ],
        "TableStatus": "ACTIVE",
        "CreationDateTime": "2021-11-28T16:08:43.236000+09:00",
        "ProvisionedThroughput": {
            "LastIncreaseDateTime": "1970-01-01T09:00:00+09:00",
            "LastDecreaseDateTime": "1970-01-01T09:00:00+09:00",
            "NumberOfDecreasesToday": 0,
            "ReadCapacityUnits": 10,
            "WriteCapacityUnits": 5
        },
        "TableSizeBytes": 0,
        "ItemCount": 0,
        "TableArn": "arn:aws:dynamodb:ddblocal:000000000000:table/Music"
    }
}

# Update a Table
$ aws dynamodb update-table --table-name Music \
    --endpoint-url http://localhost:8000 \
    --provisioned-throughput ReadCapacityUnits=20,WriteCapacityUnits=10
{
    "TableDescription": {
        "AttributeDefinitions": [
            {
                "AttributeName": "Artist",
                "AttributeType": "S"
            },
            {
                "AttributeName": "SongTitle",
                "AttributeType": "S"
            }
        ],
        "TableName": "Music",
        "KeySchema": [
            {
                "AttributeName": "Artist",
                "KeyType": "HASH"
            },
            {
                "AttributeName": "SongTitle",
                "KeyType": "RANGE"
            }
        ],
        "TableStatus": "ACTIVE",
        "CreationDateTime": "2021-11-28T16:08:43.236000+09:00",
        "ProvisionedThroughput": {
            "LastIncreaseDateTime": "1970-01-01T09:00:00+09:00",
            "LastDecreaseDateTime": "1970-01-01T09:00:00+09:00",
            "NumberOfDecreasesToday": 0,
            "ReadCapacityUnits": 20,
            "WriteCapacityUnits": 10
        },
        "TableSizeBytes": 0,
        "ItemCount": 0,
        "TableArn": "arn:aws:dynamodb:ddblocal:000000000000:table/Music"
    }
}

# Delete a Table
$ aws dynamodb update-table --table-name Music \
    --endpoint-url http://localhost:8000 \
    --billing-mode PAY_PER_REQUEST

# List Tables
$ aws dynamodb list-tables \
    --endpoint-url http://localhost:8000 
{
    "TableNames": [
        "Customer",
        "Music"
    ]
}

# Describe Provisioned Throughput Quotas
$ aws dynamodb describe-limits \
    --endpoint-url http://localhost:8000 
{
    "AccountMaxReadCapacityUnits": 80000,
    "AccountMaxWriteCapacityUnits": 80000,
    "TableMaxReadCapacityUnits": 40000,
    "TableMaxWriteCapacityUnits": 40000
}    
```

# Working with Items

## Basics

```bash
# Read an item
$ aws dynamodb get-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"1"}}'
## You can use the ConsistentRead parameter to request a strongly consistent read instead. (This consumes additional read capacity units, but it returns the most up-to-date version of the item.)
## To return the number of read capacity units consumed by GetItem, set the ReturnConsumedCapacity parameter to TOTAL
$ aws dynamodb get-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"1"}}' \
    --consistent-read \
    --projection-expression "Description, Price, RelatedItems" \
    --return-consumed-capacity TOTAL

# PutItem
## item.json
{
    "ForumName": {"S": "Amazon DynamoDB"},
    "Subject": {"S": "New discussion thread"},
    "Message": {"S": "First post in this thread"},
    "LastPostedBy": {"S": "fred@example.com"},
    "LastPostDateTime": {"S": "201603190422"}
}
$ aws dynamodb put-item \
    --endpoint-url http://localhost:8000 \
    --table-name Thread \
    --item file://item.json

# UpdateItem
## key.json
{
    "ForumName": {"S": "Amazon DynamoDB"},
    "Subject": {"S": "New discussion thread"}
}
## expression-attribute-values.json
{
    ":zero": {"N":"0"},
    ":lastpostedby": {"S":"barney@example.com"}
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name Thread \
    --key file://key.json \
    --update-expression "SET Answered = :zero, Replies = :zero, LastPostedBy = :lastpostedby" \
    --expression-attribute-values file://expression-attribute-values.json \
    --return-values ALL_NEW
# DeleteItem
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name Thread \
    --key file://key.json
```

## Bath Operations

하나의 batch 에 HTTP Request 가 포함되서 전달된다. HTTP Request 의 network round trip 을 줄일 수 있다. 게다가 모든 Operation 은 병렬로 처리된다.

```bash
# BatchGetItem
## request-items.json
{
    "Thread": {
        "Keys": [
            {
                "ForumName":{"S": "Amazon DynamoDB"},
                "Subject":{"S": "DynamoDB Thread 1"}
            },
            {
                "ForumName":{"S": "Amazon S3"},
                "Subject":{"S": "S3 Thread 1"}
            }
        ],
        "ProjectionExpression":"ForumName, Subject, LastPostedDateTime, Replies"
    }
}
$ aws dynamodb batch-get-item \
    --endpoint-url http://localhost:8000 \
    --request-items file://request-items.json

# BatchWriteItem
## request-items.json
{
    "ProductCatalog": [
        {
            "PutRequest": {
                "Item": {
                    "Id": { "N": "601" },
                    "Description": { "S": "Snowboard" },
                    "QuantityOnHand": { "N": "5" },
                    "Price": { "N": "100" }
                }
            }
        },
        {
            "PutRequest": {
                "Item": {
                    "Id": { "N": "602" },
                    "Description": { "S": "Snow shovel" }
                }
            }
        }
    ]
}
$ aws dynamodb batch-write-item \
    --endpoint-url http://localhost:8000 \
    --request-items file://request-items.json
```

## Atomic Counters

Atomicity 를 보장하고 숫자를 늘리는 방법이다.

```bash
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": { "N": "601" }}' \
    --update-expression "SET Price = Price + :incr" \
    --expression-attribute-values '{":incr":{"N":"5"}}' \
    --return-values UPDATED_NEW
```

## Conditional Writes

특별한 조건을 만족할 때만 업데이트가 되도록하는 방법이다. 
Alice 와 Bob 이 같은 Item 을 업데이트할 때 첫번째 업데이트는 성공시키고
두번째 업데이트는 실패시킨다. Consistency 를 보장한다.
Optimistic Concurrent Control 과 같다???

```bash
# Alice's expression-attribute-values.json
{
    ":newval":{"N":"8"},
    ":currval":{"N":"10"}
}
# Alice's command line
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"1"}}' \
    --update-expression "SET Price = :newval" \
    --condition-expression "Price = :currval" \
    --expression-attribute-values file://expression-attribute-values.json
# Bob's expression-attribute-values.json
{
    ":newval":{"N":"12"},
    ":currval":{"N":"10"}
}
# Bob's command line. This will be failed.
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"1"}}' \
    --update-expression "SET Price = :newval" \
    --condition-expression "Price = :currval" \
    --expression-attribute-values file://expression-attribute-values.json
```

## Specifying Item Attributes

### Nested Attributes

* `[n]` — for list elements
* `.` (dot) — for map elements

### Document Paths

* A top-level scalar attribute.
  * `Description`
* A top-level list attribute. (This returns the entire list, not just some of the elements.)
  * `RelatedItems`
* The third element from the RelatedItems list. (Remember that list elements are zero-based.)
  * `RelatedItems[2]`
* The front-view picture of the product.
  * `Pictures.FrontView`
* All of the five-star reviews.
  * `ProductReviews.FiveStar`
* The first of the five-star reviews.
  * `ProductReviews.FiveStar[0]`

## Projection Expressions

필요한 attribute 만 가져온다.

```bash
# key.json
{
    "Id": { "N": "123" }
}
# 
aws dynamodb get-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key file://key.json \
    --projection-expression "Description, RelatedItems[0], ProductReviews.FiveStar"
```

## Condition Expressions

* [Comparison Operator and Function Reference @ amazon](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.OperatorsAndFunctions.html)

----

```bash
# Put Item
# item.json
{
    "Id": {"N": "456" },
    "ProductCategory": {"S": "Sporting Goods" },
    "Price": {"N": "650" }
}
$ aws dynamodb put-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --item file://item.json

# Conditional Put
$ aws dynamodb put-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --item file://item.json \
    --condition-expression "attribute_not_exists(Id)"

# Conditional Deletes
# values.json
{
    ":cat1": {"S": "Sporting Goods"},
    ":cat2": {"S": "Gardening Supplies"},
    ":lo": {"N": "500"},
    ":hi": {"N": "600"}
}
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"456"}}' \
    --condition-expression "(ProductCategory IN (:cat1, :cat2)) and (Price between :lo and :hi)" \
    --expression-attribute-values file://values.json

# Conditional Updates
# values.json
{
    ":discount": { "N": "75"},
    ":limit": {"N": "500"}
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --update-expression "SET Price = Price - :discount" \
    --condition-expression "Price > :limit" \
    --expression-attribute-values file://values.json

# Checking for Attributes in an Item
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --condition-expression "attribute_not_exists(Price)"
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --condition-expression "attribute_exists(ProductReviews.OneStar)"

# Checking for Attribute Type
# expression-attribute-values.json
{
    ":v_sub":{"S":"SS"}
}
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --condition-expression "attribute_type(Color, :v_sub)" \
    --expression-attribute-values file://expression-attribute-values.json

# Checking String Starting Value
## expression-attribute-values.json
{
    ":v_sub":{"S":"http://"}
}
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --condition-expression "begins_with(Pictures.FrontView, :v_sub)" \
    --expression-attribute-values file://expression-attribute-values.json

# Checking for an Element in a Set
## expression-attribute-values.json
{
    ":v_sub":{"S":"Red"}
}
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --condition-expression "contains(Color, :v_sub)" \
    --expression-attribute-values file://expression-attribute-values.json

# Checking the Size of an Attribute Value
## expression-attribute-values.json
{
    ":v_sub":{"S":"Red"}
}
$ aws dynamodb delete-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id": {"N": "456"}}' \
    --condition-expression "size(VideoClip) > :v_sub" \
    --expression-attribute-values file://expression-attribute-values.json
```

## Update Expressions

* [Update Expressions](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.DELETE)

----

### SET - Modifying or Adding Item Attributes

```bash
# Put Item
## item.json
{
    "Id": {"N": "789"},
    "ProductCategory": {"S": "Home Improvement"},
    "Price": {"N": "52"},
    "InStock": {"BOOL": true},
    "Brand": {"S": "Acme"}
}
$ aws dynamodb put-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --item file://item.json

# Modifying Attributes
## values.json
{
    ":c": { "S": "Hardware" },
    ":p": { "N": "60" }
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET ProductCategory = :c, Price = :p" \
    --expression-attribute-values file://values.json \
    --return-values ALL_NEW

# Adding new Lists and Maps
## values.json
{
    ":ri": {
        "L": [
            { "S": "Hammer" }
        ]
    },
    ":pr": {
        "M": {
            "FiveStar": {
                "L": [
                    { "S": "Best product ever!" }
                ]
            }
        }
    }
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET RelatedItems = :ri, ProductReviews = :pr" \
    --expression-attribute-values file://values.json \
    --return-values ALL_NEW

# Adding Elements to a List
## values.json
{
    ":ri": { "S": "Nails" }
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET RelatedItems[1] = :ri" \
    --expression-attribute-values file://values.json \
    --return-values ALL_NEW

# Adding Nested Map Attributes
## names.json
{
    "#pr": "ProductReviews",
    "#5star": "FiveStar",
    "#3star": "ThreeStar"
}
## values.json
{
    ":r5": { "S": "Very happy with my purchase" },
    ":r3": {
        "L": [
            { "S": "Just OK - not that great" }
        ]
    }
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET #pr.#5star[1] = :r5, #pr.#3star = :r3" \
    --expression-attribute-names file://names.json \
    --expression-attribute-values file://values.json \
    --return-values ALL_NEW

# Incrementing and Decrementing Numeric Attributes
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET Price = Price - :p" \
    --expression-attribute-values '{":p": {"N":"15"}}' \
    --return-values ALL_NEW

# Appending Elements to a List
## values.json
{
    ":vals": {
        "L": [
            { "S": "Screwdriver" },
            {"S": "Hacksaw" }
        ]
    }
}
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET #ri = list_append(#ri, :vals)" \
    --expression-attribute-names '{"#ri": "RelatedItems"}' \
    --expression-attribute-values file://values.json  \
    --return-values ALL_NEW

# Appending Elements to front of a List
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET #ri = list_append(:vals, #ri)" \
    --expression-attribute-names '{"#ri": "RelatedItems"}' \
    --expression-attribute-values '{":vals": {"L": [ { "S": "Chisel" }]}}' \
    --return-values ALL_NEW

# Preventing Overwrites of an Existing Attribute
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "SET Price = if_not_exists(Price, :p)" \
    --expression-attribute-values '{":p": {"N": "100"}}' \
    --return-values ALL_NEW
```

### REMOVE - Deleting Attributes from an Item

```bash
# Just remove
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "REMOVE Brand, InStock, QuantityOnHand" \
    --return-values ALL_NEW

# Removing Elements from a List
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "REMOVE RelatedItems[1], RelatedItems[2]" \
    --return-values ALL_NEW
```

### ADD - Updating Numbers and Sets

```bash
# Adding a Number
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "ADD QuantityOnHand :q" \
    --expression-attribute-values '{":q": {"N": "5"}}' \
    --return-values ALL_NEW

# Adding Elements to a Set
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "ADD Color :c" \
    --expression-attribute-values '{":c": {"SS":["Orange", "Purple"]}}' \
    --return-values ALL_NEW
# Now Color exists, Add more elements
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "ADD Color :c" \
    --expression-attribute-values '{":c": {"SS":["Yellow", "Green", "Blue"]}}' \
    --return-values ALL_NEW
```

### DELETE - Removing Elements from a Set

```bash
$ aws dynamodb update-item \
    --endpoint-url http://localhost:8000 \
    --table-name ProductCatalog \
    --key '{"Id":{"N":"789"}}' \
    --update-expression "DELETE Color :p" \
    --expression-attribute-values '{":p": {"SS": ["Yellow", "Purple"]}}' \
    --return-values ALL_NEW
```

# Working with Queries

## Key Condition Expressions

* `a = b` — true if the attribute a is equal to the value b
* `a < b` — true if a is less than b
* `a <= b` — true if a is less than or equal to b
* `a > b` — true if a is greater than b
* `a >= b` — true if a is greater than or equal to b
* `a BETWEEN b AND c` — true if a is greater than or equal to b, and less than or equal to c.

```bash
## values.json
{
    ":name":{"S":"Amazon DynamoDB"},
    ":sub":{"S":"DynamoDB Thread 1"}
}
$ aws dynamodb query \
    --endpoint-url http://localhost:8000 \
    --table-name Thread \
    --key-condition-expression "ForumName = :name and Subject = :sub" \
    --expression-attribute-values  file://values.json

## values.json
{
    ":id":{"S":"Amazon DynamoDB#DynamoDB Thread 1"},
    ":dt":{"S":"2015-09"}
}
$ aws dynamodb query \
    --endpoint-url http://localhost:8000 \
    --table-name Reply \
    --key-condition-expression "Id = :id and begins_with(ReplyDateTime, :dt)" \
    --expression-attribute-values  file://values.json
```

## Filter Expressions for Query

filter 는 query result 에 적용된다. 즉, query 가 DynamoDB 에서 실행되고 client 로
result 가 도착했을 때 filter 가 적용된다.

```bash
## values.json
{
    ":fn":{"S":"Amazon DynamoDB"},
    ":sub":{"S":"DynamoDB Thread 1"},
    ":num":{"N":"3"}
}
$ aws dynamodb query \
    --endpoint-url http://localhost:8000 \
    --table-name Thread \
    --key-condition-expression "ForumName = :fn and Subject = :sub" \
    --filter-expression "#v >= :num" \
    --expression-attribute-names '{"#v": "Views"}' \
    --expression-attribute-values file://values.json
```

## Paginating Table Query Results

* [Paginating Table Query Results @ amazon](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.Pagination.html)

----

```bash
$ aws dynamodb query --table-name Movies \
    --endpoint-url http://localhost:8000 \
    --projection-expression "title" \
    --key-condition-expression "#y = :yyyy" \
    --expression-attribute-names '{"#y":"year"}' \
    --expression-attribute-values '{":yyyy":{"N":"1993"}}' \
    --page-size 5 \
    --debug
2017-07-07 11:13:15,603 - MainThread - botocore.parsers - DEBUG - Response body:
b'{"Count":5,"Items":[{"title":{"S":"A Bronx Tale"}},
{"title":{"S":"A Perfect World"}},{"title":{"S":"Addams Family Values"}},
{"title":{"S":"Alive"}},{"title":{"S":"Benny & Joon"}}],
"LastEvaluatedKey":{"year":{"N":"1993"},"title":{"S":"Benny & Joon"}},
"ScannedCount":5}'
```

다음은 위의 결과를 pretty print 한 것이다.

```json
{
  "Count": 5,
  "Items": [
    {
      "title": {
        "S": "A Bronx Tale"
      }
    },
    {
      "title": {
        "S": "A Perfect World"
      }
    },
    {
      "title": {
        "S": "Addams Family Values"
      }
    },
    {
      "title": {
        "S": "Alive"
      }
    },
    {
      "title": {
        "S": "Benny & Joon"
      }
    }
  ],
  "LastEvaluatedKey": {
    "year": {
      "N": "1993"
    },
    "title": {
      "S": "Benny & Joon"
    }
  },
  "ScannedCount": 5
}
```

다음 페이지는 LastEvaluatedKey 를 채워서 보냄???

# Working with Scans

## Filter Expressions for Scan 

```bash
$ aws dynamodb scan \
    --endpoint-url http://localhost:8000 \
     --table-name Thread \
     --filter-expression "LastPostedBy = :name" \
     --expression-attribute-values '{":name":{"S":"User A"}}'
```

## Paginating the Results

```bash
$ aws dynamodb scan \
    --table-name Movies \
    --projection-expression "title" \
    --filter-expression 'contains(info.genres,:gen)' \
    --expression-attribute-values '{":gen":{"S":"Sci-Fi"}}' \
    --page-size 100  \
    --debug
2017-07-07 12:19:14,389 - MainThread - botocore.parsers - DEBUG - Response body:
b'{"Count":7,"Items":[{"title":{"S":"Monster on the Campus"}},{"title":{"S":"+1"}},
{"title":{"S":"100 Degrees Below Zero"}},{"title":{"S":"About Time"}},{"title":{"S":"After Earth"}},
{"title":{"S":"Age of Dinosaurs"}},{"title":{"S":"Cloudy with a Chance of Meatballs 2"}}],
"LastEvaluatedKey":{"year":{"N":"2013"},"title":{"S":"Curse of Chucky"}},"ScannedCount":100}'

# The reponse of the last page. There is no LastEvaluatedKey
2017-07-07 12:19:17,830 - MainThread - botocore.parsers - DEBUG - Response body:
b'{"Count":1,"Items":[{"title":{"S":"WarGames"}}],"ScannedCount":6}'
```

## Counting the Items in the Results

* **ScannedCount** — The number of items evaluated, before any ScanFilter is applied. A high ScannedCount value with few, or no, Count results indicates an inefficient Scan operation. If you did not use a filter in the request, ScannedCount is the same as Count.
* **Count** — The number of items that remain, after a filter expression (if present) was applied.

# Working with Transactions

## TransactWriteItems API

25 개 까지의 write action (Put, Update, Delete, Condition Check) 들을 atomicity (all-or-nothing) 을 보장해준다.

Client Token 을 이용하면 Idempotency 가 보장된다.

## TransactGetItems API

25 개 까지의 read action (Get) 들의 synchronous 을 보장해준다. 읽어온 결과는 4MB 를 초과할 수는
없다. 

## Isolation Levels for DynamoDB Transactions

Operation Summary. `*` 은 해당 Isolation Level 이 unit 으로 적용될 때를 말한다.
개별 operation 의 Isolation Level 은 serializable 하다.

| Operation | Isolation Level |
|---|---|
| DeleteItem |  Serializable |
| PutItem  | Serializable  |
| UpdateItem  | Serializable  |
| GetItem |  Serializable |
| BatchGetItem  | Read-committed*  |
| BatchWriteItem  | NOT Serializable*  |
| Query  |  Read-committed* |
| Scan  | Read-committed*  |
| Other transactional operation  | Serializable  |

## Transaction Conflict Handling in DynamoDB

* TransactWriteItems 진행중인 item 에 대하여 **PutItem**, **UpdateItem** 혹은 **DeleteItem** 을 요청했을 때
* TransactWriteItems 진행중인 item 에 대하여 **TransactWriteItems** 을 요청했을 때
* TransactWriteItems, BatchWriteItem, PutItem, UpdateItem 혹은 DeleteItem 진행중인 item 에 대하여 **TransactGetItems** 을 요청했을 때

## Best Practices for Transactions

## Using Transactional APIs with Global Tables

하나의 Table 을 여러 region 에 Replication 하는 것을 Global Tables 이라 한다. 
Global Tables 을 이용하는 경우는 하나의 Region 에서 TransactWriteItems 를 수행하면
다른 Region 으로 propagation 되는 방식이다.

예를 들어 Global Tables 를 US East (Ohio), US West (Oregon) regions 에 운영한다고 해보자.
US East (Ohio) 에서 TransactWriteItems 를 수행한다고 하자. 그 Transaction 이 commit 되야
US West (Oregon) 에 replication 된다.
