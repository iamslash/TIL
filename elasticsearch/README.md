- [Materials](#materials)
- [Install Elasticsearch 2.4.1 from the source](#install-elasticsearch-241-from-the-source)
- [Install with Docker](#install-with-docker)
- [Sample Data](#sample-data)
- [Integration with Spring](#integration-with-spring)
- [Basic](#basic)
  - [Elastic \_cat api](#elastic-_cat-api)
  - [Elastic Search vs RDBMS](#elastic-search-vs-rdbms)
  - [CRUD](#crud)
  - [Update](#update)
  - [Bulk](#bulk)
  - [Mapping](#mapping)
  - [Elasticsearch 쿼리 전체 종류 (기본적인 분류)](#elasticsearch-쿼리-전체-종류-기본적인-분류)
    - [1. **Term-Level Queries** (정확한 값 일치)](#1-term-level-queries-정확한-값-일치)
    - [2. **Full-Text Queries** (자연어 검색용, 분석기 적용)](#2-full-text-queries-자연어-검색용-분석기-적용)
    - [3. **Compound Queries** (복합 쿼리 - 쿼리들을 조합)](#3-compound-queries-복합-쿼리---쿼리들을-조합)
    - [4. **Span Queries** (고급 문장 내 위치 기반 검색)](#4-span-queries-고급-문장-내-위치-기반-검색)
    - [5. **Joining Queries** (관계형 구조 흉내)](#5-joining-queries-관계형-구조-흉내)
    - [6. **Geo Queries** (위치 기반)](#6-geo-queries-위치-기반)
    - [7. **Specialized / Other Queries**](#7-specialized--other-queries)
  - [🎯 실무에서 가장 자주 쓰는 조합](#-실무에서-가장-자주-쓰는-조합)
  - [Examples](#examples)
  - [Search](#search)
  - [Metric Aggregation](#metric-aggregation)
  - [Bucket Aggregation](#bucket-aggregation)
  - [Sub Aggregation](#sub-aggregation)
  - [Pipeline Aggregation](#pipeline-aggregation)
- [Plugins](#plugins)
  - [kopf](#kopf)
- [Advanced](#advanced)
  - [Cluster settings for concurrent rebalance](#cluster-settings-for-concurrent-rebalance)
  - [Delete old indices](#delete-old-indices)
  - [Adding and removing a node](#adding-and-removing-a-node)
  - [Rolling Restart](#rolling-restart)
  - [Rolling Upgrade](#rolling-upgrade)
  - [Open vs close Index](#open-vs-close-index)
  - [Reindex](#reindex)
  - [`refresh_interval` from Index Setting](#refresh_interval-from-index-setting)

----

# Materials

* [ElasticSearch Cheat Sheet 2.x, 7.x](https://elasticsearch-cheatsheet.jolicode.com/#es2)
* [Spring and Elasticsearch @ github](https://github.com/ahnjunwoo/elasticsearchDockerExam)
* [elk @ TIL](/elk/README.md)
* [elasticsearch 실습 정리 @ github](https://github.com/itmare/es)
* [Elastic 가이드 북](https://esbook.kimjmin.net/)

# Install Elasticsearch 2.4.1 from the source

This is matched with Kibana 4.6.6. You can download from [Elastic past release](https://www.elastic.co/kr/downloads/past-releases).

```bash
$ wget https://download.elastic.co/elasticsearch/release/org/elasticsearch/distribution/tar/elasticsearch/2.4.1/elasticsearch-2.4.1.tar.gz
$ tar xzvf elasticsearch-2.4.1.tar.gz
$ cd elasticsearch-2.4.1

$ vim config/elasticsearch.yml
cluster.name: iamslash-es
node.name: node-1
path.data: /iamslash/service/es/data
path.logs: /iamslash/logs/es
network.host: _eth0_
http.port: 19200
mapper.allow_dots_in_name: true

$ nohup /usr/bin/java -Xms256m -Xmx1g -Djava.awt.headless=true -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=75 -XX:+UseCMSInitiatingOccupancyOnly -XX:+HeapDumpOnOutOfMemoryError -XX:+DisableExplicitGC -Dfile.encoding=UTF-8 -Djna.nosys=true -Dmapper.allow_dots_in_name=true -Des.path.home=/home/iamslash/elasticsearch-2.4.1 -cp /home/iamslash/elasticsearch-2.4.1/lib/elasticsearch-2.4.1.jar:/home/iamslash/elasticsearch-2.4.1/lib/* org.elasticsearch.bootstrap.Elasticsearch start -d & 
# After exit terminal, systemd will be a parent process of /usr/bin/java
```

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

## Open, close index
# Close a specific index
curl -XPOST 'localhost:9200/my_index/_close'
# Open a specific index
curl -XPOST 'localhost:9200/my_index/_open'
# Close all indices
curl -XPOST 'localhost:9200/*/_close'
# Open all indices
curl -XPOST 'localhost:9200/_all/_close'

## delete index classes
$ curl -XDELETE 'http://localhost:9200/classes?pretty'
{
  "acknowledged" : true
}
# Delete all indices
$ curl -XDELETE 'http://localhost:9200/_all/'
$ curl -XDELETE 'http://localhost:9200/*/'

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

## Elasticsearch 쿼리 전체 종류 (기본적인 분류)

[ElasticSearch Query](elasticsearch_query.md)

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

* [입문 6장 ELASTICSEARCH -분석 & 집계](https://kazaana2009.tistory.com/7)
* [8.1 메트릭 - Metrics Aggregations](https://esbook.kimjmin.net/08-aggregations/8.1-metrics-aggregations)

-----

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

* [입문 6장 ELASTICSEARCH -분석 & 집계](https://kazaana2009.tistory.com/7)
* [8.2 버킷 - Bucket Aggregations](https://esbook.kimjmin.net/08-aggregations/8.2-bucket-aggregations)

----

Bucket Aggregation 은 RDBMS 의 group by 와 비슷하다. document 를 group 으로 묶는다.

```bash
$ curl -H 'Content-type: application/json' -XPOST http://localhost:9200/_bulk?pretty --data-binary @twoteam_basketball.json

# 안된다???
$ curl -H 'Content-type: application/json' -XGET http://localhost:9200/_search?pretty --data-binary @stats_by_team.json
```

## Sub Aggregation

* [8.3 하위 - sub-aggregations](https://esbook.kimjmin.net/08-aggregations/8.3-aggregations)

## Pipeline Aggregation

* [8.4 파이프라인 - Pipeline Aggregations](https://esbook.kimjmin.net/08-aggregations/8.4-pipeline-aggregations)

# Plugins

## kopf

* [web admin interface for elasticsearch](https://github.com/lmenezes/elasticsearch-kopf)
  * 킹왕짱 web admin
-----

```bash
$ ./elasticsearch/bin/plugin install lmenezes/elasticsearch-kopf/{branch|version}
$ open http://localhost:9200/_plugin/kopf
```

# Advanced

## Cluster settings for concurrent rebalance 

* [Cluster-level shard allocation and routing settings](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-cluster.html)

----

* `cluster.routing.allocation.cluster_concurrent_rebalance`
  * total allowed count of shards for concurrent rebalance
* `cluster.routing.allocation.node_concurrent_recoveries`


## Delete old indices

* [curator @ github](https://github.com/elastic/curator)
  * [install curator @ ES](https://www.elastic.co/guide/en/elasticsearch/client/curator/current/installation.html)
  * [curator reference @ ES](https://www.elastic.co/guide/en/elasticsearch/client/curator/current/index.html)

-----

```console
curator --host <IP> delete indices --older-than 30 --prefix "twitter-" --time-unit days  --timestring '%Y-%m-%d'
```

## Adding and removing a node

* [Shar Allocation Filtering @ ES](https://www.elastic.co/guide/en/elasticsearch/reference/2.4/allocation-filtering.html)

-----

* Delete old indices as much as possible to save time for rebalancing.
* Provision a new node.
* Attach a new node and check all shard are rebalanced.
* Attach a new node to Load Balancer.
* Detach a old node from Load Balancer.
* Exclude a old node and check all shards are rebalanced.
* Shutdown a old node.

## Rolling Restart

* [[Elasticsearch] 클러스터 rolling restarts](https://lng1982.tistory.com/315)

## Rolling Upgrade

* [Rolling Upgrades @ ES](https://www.elastic.co/guide/en/elasticsearch/reference/current/rolling-upgrades.html)

## Open vs close Index

* [Open / Close Index API](https://www.elastic.co/guide/en/elasticsearch/reference/6.8/indices-open-close.html)
* [[elasticsearch] open / close /delete index](https://kugancity.tistory.com/entry/elasticsearch-delete-open-close-index)

-----

A closed index is blocked for read/write. A closed index can be opened which will then go through the normal recovery process.

If you want to reduce the overhead of an index while keeping it available for occasional searches, **freeze** the index instead. If you want to store an index outside of the cluster, use a **snapshot**.

```bash
# Close a specific index
curl -XPOST 'localhost:9200/my_index/_close'
# Open a specific index
curl -XPOST 'localhost:9200/my_index/_open'
# Close all indices
curl -XPOST 'localhost:9200/*/_close'
# Open all indices
curl -XPOST 'localhost:9200/_all/_close'
```

## Reindex

* [ElasticSearch 에서 reindex 을 활용하는 방법](https://findstar.pe.kr/2018/07/07/elasticsearch-reindex/)
* [Reindex API @ Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-reindex.html)
* A Cluster 에서 A Cluster 혹은 A Cluster 에서 B Cluster 로 index 를 이동할 때 사용한다. 

-----

* Reindex from local.

```bash
$ curl -X POST "localhost:9200/_reindex?pretty" -H 'Content-Type: application/json' -d'
{
  "source": {
    "index": "my-index-000001"
  },
  "dest": {
    "index": "my-new-index-000001"
  }
}
'
```

* Reindex from remote.

```bash
curl -X POST "localhost:9200/_reindex?pretty" -H 'Content-Type: application/json' -d'
{
  "source": {
    "remote": {
      "host": "http://otherhost:9200",
      "username": "user",
      "password": "pass"
    },
    "index": "my-index-000001",
    "query": {
      "match": {
        "test": "data"
      }
    }
  },
  "dest": {
    "index": "my-new-index-000001"
  }
}
'
```

## `refresh_interval` from Index Setting

✅ 정의

`"refresh_interval": "1s"`

이 설정은 Elasticsearch가 세그먼트를 디스크에 반영하고 검색 가능 상태로 전환하는 시간 간격을 의미합니다.

기본값은 "1s" (1초마다 refresh가 수행됨)

✅ 어떻게 동작하나?

문서를 색인 (index)하면, 그 문서는 바로 검색 가능한 것은 아님 refresh_interval이 지나고 나면, Lucene 세그먼트가 flush되고, 문서가 검색 대상에 포함됨.

✅ 예시

`refresh_interval: 1s`

→ 문서를 색인하고 약 1초 후 검색 가능

`refresh_interval: 30s`

→ 색인 후 30초가 지나야 검색 가능

`refresh_interval: -1`

→ 자동 refresh를 끄고, 수동으로 `POST /<index>/_refresh` 요청해야만 검색 가능

✅ 실무 팁

| 상황	| 추천 설정 |
|--|--|
| 실시간 검색이 필요한 경우 (예: 채팅, 검색엔진) |	`1s` (기본값 유지) |
|대량 색인 작업 중 (성능 최적화)	| `"refresh_interval": "-1"` 후 일괄 색인 후 `_refresh` 호출 |
| 색인 지연이 허용되는 경우 (예: 로그 저장)	| `30s` 이상으로 설정 가능 |

✅ 참고 사항

refresh는 리소스를 사용하는 작업입니다 (디스크/메모리).

너무 짧게 설정하면 성능에 영향을 줄 수 있고,
너무 길게 설정하면 검색 지연이 발생할 수 있습니다.
