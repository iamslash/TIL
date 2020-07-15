# Materials

* [DynamoDB @ pyrasis](http://pyrasis.com/book/TheArtOfAmazonWebServices/Chapter30/09)

# Install with docker

* [amazon/dynamodb-local
](https://hub.docker.com/r/amazon/dynamodb-local/)

```console
$ docker pull amazon/dynamodb-local
$ docker run -d -p 8000:8000 --rm --name my-dynamodb amazon/dynamodb-local
```

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
