- [Materials](#materials)
- [Install with Docker](#install-with-docker)
- [Sample Data](#sample-data)
- [Integration with Spring](#integration-with-spring)
- [Basic](#basic)
  - [Elastic _cat api](#elastic-_cat-api)
  - [Elastic Search vs RDBMS](#elastic-search-vs-rdbms)
  - [CRUD](#crud)
  - [Update](#update)
  - [Bulk](#bulk)
  - [Mapping](#mapping)
  - [Search](#search)
  - [Metric Aggregation](#metric-aggregation)
  - [Bucket Aggregation](#bucket-aggregation)

----

# Materials

* [Spring and Elasticsearch @ github](https://github.com/ahnjunwoo/elasticsearchDockerExam)
* [elk @ TIL](/elk/README.md)

# Install with Docker

```console
$ git clone git@github.com:ahnjunwoo/elasticsearchDockerExam.git
$ cd elasticsearchDockerExam
$ docker-compose up -d

$ curl localhost:9200/_cat 
```

```console
$ docker run -d --rm --name my-es -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2

$ curl localhost:9200/_cat
```

# Sample Data

* [샘플 데이터 로드](https://www.elastic.co/guide/kr/kibana/current/tutorial-load-dataset.html)
  * 샘플 데이터를 다운로드해서 bulk 로 입력할 수 있다.

```bash
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/bank/account/_bulk?pretty' --data-binary @accounts.json
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/shakespeare/_bulk?pretty' --data-binary @shakespeare.json
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/_bulk?pretty' --data-binary @logs.jsonl
```

# Integration with Spring

* [spring-examples/exelasticsearch @ github](https://github.com/iamslash/spring-examples/blob/master/exelasticsearch/README.md)

# Basic

## Elastic _cat api

* [[elasticsearch5] cat API 간단 설명과 유용한 cat API 예제](https://knight76.tistory.com/entry/elasticsearch-5-cat-API)

-----

```bash
$ curl -X GET http://localhost:9200/_cat

# show health
$ curl -XGET 'localhost:9200/_cat/health'

# show health with columns
$ curl -XGET 'localhost:9200/_cat/health?v'

# show headers of health
$ curl -XGET 'localhost:9200/_cat/health?help'

# show specific headers of health
$ curl -XGET 'localhost:9200/_cat/health?h=cluster,status'

# show indices
$ curl -XGET 'localhost:9200/_cat/indices'

# show indices with bytes
$ curl -XGET 'localhost:9200/_cat/indices?bytes=b'

# show master node
$ curl -XGET 'localhost:9200/_cat/master?v'

# show name, role, load, uptime of nodes
$ curl -XGET 'localhost:9200/_cat/nodes?v&h=name,node.role,load,uptime'
```

## Elastic Search vs RDBMS

| Elastic Search | RDBMS |
|:---------------|:------|
| Index          | Database |
| Type | Table |
| Document | Row |
| Field | Column |
| Mapping | Schema |

| Elastic Search | RDBMS |
|:---------------|:------|
| GET | SELECT |
| POST | UPDATE |
| PUT | INSERT |
| DELETE | DELETE |

```bash
$ curl -XGET localhost:9200/classes/class/1
# SELECT * FROM class WHERE id = 1

$ curl -XPOST localhost:9200/classes/class/1 -d '{XXX}'
# INSERT * INTO class VALUES(XXX)
$ curl -XPOST localhost:9200/classes/class/1 -d '{XXX}'
# UPDATE class SET XXX WHERE id = 1

$ curl -XDELTE localhost:9200/classes/class/1
# DELETE FROM class WHERE id = 1
```

## CRUD

```bash

## try to get something
$ curl -XGET http://localhost:9200/classes
{
  "error" : {
    "root_cause" : [
      {
        "type" : "index_not_found_exception",
        "reason" : "no such index [classes]",
        "resource.type" : "index_or_alias",
        "resource.id" : "classes",
        "index_uuid" : "_na_",
        "index" : "classes"
      }
    ],
    "type" : "index_not_found_exception",
    "reason" : "no such index [classes]",
    "resource.type" : "index_or_alias",
    "resource.id" : "classes",
    "index_uuid" : "_na_",
    "index" : "classes"
  },
  "status" : 404
}

## create index classes
$ curl -XPUT http://localhost:9200/classes?pretty
{
  "acknowledged" : true,
  "shards_acknowledged" : true,
  "index" : "classes"
}

## delete index classes
curl -XDELETE http://localhost:9200/classes?pretty
{
  "acknowledged" : true
}

## create document but fail
curl -XPUT http://localhost:9200/classes/class/1?pretty -d '{"title": "Algorithm", "professor": "John"}'
{
  "error" : "Content-Type header [application/x-www-form-urlencoded] is not supported",
  "status" : 406
}

## create document with header
curl -H 'Content-Type: application/json' -XPUT http://localhost:9200/classes/class/1?pretty -d '{"title": "Algorithm", "professor": "John"}'
{
  "_index" : "classes",
  "_type" : "class",
  "_id" : "1",
  "_version" : 1,
  "result" : "created",
  "_shards" : {
    "total" : 2,
    "successful" : 1,
    "failed" : 0
  },
  "_seq_no" : 0,
  "_primary_term" : 1
}

## create document with file
curl -H 'Content-type: application/json' -XPUT http://localhost:9200/classes/class/1?pretty -d @a.json

a.json
{"title": "Programming Language", "professor": "Tom"}

{
  "_index" : "classes",
  "_type" : "class",
  "_id" : "1",
  "_version" : 2,
  "result" : "updated",
  "_shards" : {
    "total" : 2,
    "successful" : 1,
    "failed" : 0
  },
  "_seq_no" : 1,
  "_primary_term" : 2
}
```

## Update

```bash
## update document 1
$ curl -H 'Content-type: application/json' -XPOST http://localhost:9200/classes/class/1/_update?pretty -d '{"doc":{"unit":1}}'
# {
#   "_index" : "classes",
#   "_type" : "class",
#   "_id" : "1",
#   "_version" : 3,
#   "result" : "updated",
#   "_shards" : {
#     "total" : 2,
#     "successful" : 1,
#     "failed" : 0
#   },
#   "_seq_no" : 2,
#   "_primary_term" : 2
# }

## get document 1
$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/classes/class/1?pretty
# {
#   "_index" : "classes",
#   "_type" : "class",
#   "_id" : "1",
#   "_version" : 3,
#   "_seq_no" : 2,
#   "_primary_term" : 2,
#   "found" : true,
#   "_source" : {
#     "title" : "Programming Language",
#     "professor" : "Tome",
#     "unit" : 1
#   }
# }

## update with script
$ curl -H 'Content-type: application/json' -XPOST http://localhost:9200/classes/class/1/_update?pretty -d '{"script":"ctx._source.unit += 5"}'
# {
#   "_index" : "classes",
#   "_type" : "class",
#   "_id" : "1",
#   "_version" : 4,
#   "result" : "updated",
#   "_shards" : {
#     "total" : 2,
#     "successful" : 1,
#     "failed" : 0
#   },
#   "_seq_no" : 3,
#   "_primary_term" : 2
# }

## get document 1
$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/classes/class/1?pretty
# {
#   "_index" : "classes",
#   "_type" : "class",
#   "_id" : "1",
#   "_version" : 4,
#   "_seq_no" : 3,
#   "_primary_term" : 2,
#   "found" : true,
#   "_source" : {
#     "title" : "Programming Language",
#     "professor" : "Tome",
#     "unit" : 6
#   }
# }
```

## Bulk

한번에 여러개의 document 를 삽입하는 방법

```bash
curl -H 'Content-type: application/json' -XPOST http://localhost:9200/_bulk?pretty --data-binary @classes.json
curl -H 'Content-type: application/json' -XGET http://localhost:9200/classes/class/1?pretty
curl -H 'Content-type: application/json' -XGET http://localhost:9200/classes/class/2?pretty
```

## Mapping

RDBMS 의 schema 와 같다. 필드의 타입이 정해져야 kibana 에서 시각화 할 때 용이하다. 예를 들어 필드의 타입이 정해지지 않으면 날짜 데이터가가 문자열로 저장된다.

```bash
## put maping 그러나 elasticsearch 8.0 에서 안된다. bulk 로 입력하면 이미 mapping 이 설정되어 있다.
$ curl -H 'Content-type: application/json' -XPUT 'http://localhost:9200/classes/_mapping?pretty' -d @classesRating_mapping.json
# classesRating_mapping.json
# {
#         "class" : {
#                 "properties" : {
#                         "title" : {
#                                 "type" : "string"
#                         },
#                         "professor" : {
#                                 "type" : "string"
#                         },
#                         "major" : {
#                                 "type" : "string"
#                         },
#                         "semester" : {
#                                 "type" : "string"
#                         },
#                         "student_count" : {
#                                 "type" : "integer"
#                         },
#                         "unit" : {
#                                 "type" : "integer"
#                         },
#                         "rating" : {
#                                 "type" : "integer"
#                         },
#                         "submit_date" : {
#                                 "type" : "date",
#                                 "format" : "yyyy-MM-dd"
#                         },
#                         "school_location" : {
#                                 "type" : "geo_point"
#                         }
#                 }
#         }
# }
#
#
# {
#   "error" : {
#     "root_cause" : [
#       {
#         "type" : "mapper_parsing_exception",
#         "reason" : "No handler for type [string] declared on field [professor]"
#       }
#     ],
#     "type" : "mapper_parsing_exception",
#     "reason" : "No handler for type [string] declared on field [professor]"
#   },
#   "status" : 400
# }
```

## Search

```bash
$ curl -H 'Content-type: application/json' -XPOST http://localhost:9200/_bulk?pretty --data-binary @simple_basketball.json

$ curl -H 'Content-type: application/json' -XGET 'http://localhost:9200/basketball/record/_search?pretty'
{
  "took" : 2,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 2,
      "relation" : "eq"
    },
    "max_score" : 1.0,
    "hits" : [
      {
        "_index" : "basketball",
        "_type" : "record",
        "_id" : "1",
        "_score" : 1.0,
        "_source" : {
          "team" : "Chicago Bulls",
          "name" : "Michael Jordan",
          "points" : 30,
          "rebounds" : 3,
          "assists" : 4,
          "submit_date" : "1996-10-11"
        }
      },
      {
        "_index" : "basketball",
        "_type" : "record",
        "_id" : "2",
        "_score" : 1.0,
        "_source" : {
          "team" : "Chicago Bulls",
          "name" : "Michael Jordan",
          "points" : 20,
          "rebounds" : 5,
          "assists" : 8,
          "submit_date" : "1996-10-11"
        }
      }
    ]
  }
}

# search with uri
$ curl -H 'Content-type: application/json' -XGET 'http://localhost:9200/basketball/record/_search?q=points:30&pretty'
{
  "took" : 15,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 1,
      "relation" : "eq"
    },
    "max_score" : 1.0,
    "hits" : [
      {
        "_index" : "basketball",
        "_type" : "record",
        "_id" : "1",
        "_score" : 1.0,
        "_source" : {
          "team" : "Chicago Bulls",
          "name" : "Michael Jordan",
          "points" : 30,
          "rebounds" : 3,
          "assists" : 4,
          "submit_date" : "1996-10-11"
        }
      }
    ]
  }
}

# search with request body
$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/basketball/record/_search?pretty -d '{"query": {"term": {"points": 30}}}'
```

## Metric Aggregation

aggregation 은 document 의 field 들을 조합하여 어떠한 값을 도출하는 방법이다.
Metric Aggregation 은 평균, 최소, 최대값과 같은 산술연산을 통해 조합하는 방법이다.

```bash
$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/_search?pretty --data-binary @avg_points_aggs.json
# avg_points_aggs.json
# {
#         "size" : 0,
#         "aggs" : {
#                 "avg_score" : {
#                         "avg" : {
#                                 "field" : "points"
#                         }
#                 }
#         }
# }

$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/_search?pretty --data-binary @max_points_aggs.json
# max_points_aggs.json
# {
#         "size" : 0,
#         "aggs" : {
#                 "max_score" : {
#                         "max" : {
#                                 "field" : "points"
#                         }
#                 }
#         }
# }

$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/_search?pretty --data-binary @min_points_aggs.json
# min_points_aggs.json
# {
#         "size" : 0,
#         "aggs" : {
#                 "min_score" : {
#                         "min" : {
#                                 "field" : "points"
#                         }
#                 }
#         }
# }

$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/_search?pretty --data-binary @min_points_aggs.json
# stats_points_aggs.json
# {
#         "size" : 0,
#         "aggs" : {
#                 "stats_score" : {
#                         "stats" : {
#                                 "field" : "points"
#                         }
#                 }
#         }
# }
```

## Bucket Aggregation

Bucket Aggregation 은 RDBMS 의 group by 와 비슷하다. document 를 group 으로 묶는다.

```bash
$ curl -H 'Content-type: application/json' -XPOST http://localhost:9200/_bulk?pretty --data-binary @twoteam_basketball.json

# 안된다???
$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/_search?pretty --data-binary @stats_by_team.json
```