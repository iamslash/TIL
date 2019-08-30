# Abstract

OLAP (Online Analytical Processing) soution 중 하나인 ELK 에 대해 정리한다.

# Materials

* [ELK 스택 (ElasticSearch, Logstash, Kibana) 으로 데이터 분석 @ inflearn](https://www.inflearn.com/course/elk-%EC%8A%A4%ED%83%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D/lecture/5506)
  * 킹왕짱
  * [src](https://github.com/minsuk-heo/BigData)
* [Elasticsearch, Logstash, Kibana (ELK) Docker image documentation](https://elk-docker.readthedocs.io/)
  * elk docker image
  * [image](https://hub.docker.com/r/sebp/elk)

# Install on docker

* [Elasticsearch, Logstash, Kibana (ELK) Docker image](https://hub.docker.com/r/sebp/elk) docker image 를 설치한다.

```bash
docker search elk
docker pull sebp/elk
docker run -p 5601:5601  -p 9200:9200 -p 5000:5000 -it --name my-elk sebp/elk
docker exec -it my-elk /bin/bash
```

* browser 를 이용하여 `localhost:5601` (kibana) 에 접속한다.
* browser 를 이용하여 `localhost:9200` (elasticsearch) 에 접속한다.

# Basic Elastic Search

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
curl -XGET localhost:9200/classes/class/1
# SELECT * FROM class WHERE id = 1
curl -XPOST localhost:9200/classes/class/1 -d '{XXX}'
# INSERT * INTO class VALUES(XXX)
curl -XPOST localhost:9200/classes/class/1 -d '{XXX}'
# UPDATE class SET XXX WHERE id = 1
curl -XDELTE localhost:9200/classes/class/1
# DELETE FROM class WHERE id = 1
```

## CRUD

```bash

## try to get something
curl -XGET http://localhost:9200/classes
# {
#   "error" : {
#     "root_cause" : [
#       {
#         "type" : "index_not_found_exception",
#         "reason" : "no such index [classes]",
#         "resource.type" : "index_or_alias",
#         "resource.id" : "classes",
#         "index_uuid" : "_na_",
#         "index" : "classes"
#       }
#     ],
#     "type" : "index_not_found_exception",
#     "reason" : "no such index [classes]",
#     "resource.type" : "index_or_alias",
#     "resource.id" : "classes",
#     "index_uuid" : "_na_",
#     "index" : "classes"
#   },
#   "status" : 404
# }

## create index classes
curl -XPUT http://localhost:9200/classes?pretty
# {
#   "acknowledged" : true,
#   "shards_acknowledged" : true,
#   "index" : "classes"
# }

## delete index classes
curl -XDELETE http://localhost:9200/classes?pretty
# {
#   "acknowledged" : true
# }

## create document but fail
curl -XPUT http://localhost:9200/classes/class/1?pretty -d '{"title": "Algorithm", "professor": "John"}'
# {
#   "error" : "Content-Type header [application/x-www-form-urlencoded] is not supported",
#   "status" : 406
# }

## create document with header
curl -H 'Content-Type: application/json' -XPUT http://localhost:9200/classes/class/1?pretty -d '{"title": "Algorithm", "professor": "John"}'
# {
#   "_index" : "classes",
#   "_type" : "class",
#   "_id" : "1",
#   "_version" : 1,
#   "result" : "created",
#   "_shards" : {
#     "total" : 2,
#     "successful" : 1,
#     "failed" : 0
#   },
#   "_seq_no" : 0,
#   "_primary_term" : 1
# }

## create document with file
curl -H 'Content-type: application/json' -XPUT http://localhost:9200/classes/class/1?pretty -d @a.json
#
# a.json
# {"title": "Programming Language", "professor": "Tom"}
#
# {
#   "_index" : "classes",
#   "_type" : "class",
#   "_id" : "1",
#   "_version" : 2,
#   "result" : "updated",
#   "_shards" : {
#     "total" : 2,
#     "successful" : 1,
#     "failed" : 0
#   },
#   "_seq_no" : 1,
#   "_primary_term" : 2
# }

```

## Update

```bash
```

## Bulk

```bash
```

## Mapping

```bash
```

## Search

```bash
```

## Metric Aggregation

```bash
```

## Bucket Aggreation

```bash
```

# Basic Kibana

# Basic Logstash