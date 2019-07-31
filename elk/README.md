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

# Basic Elastic Search

## Elastic Search vs RDBMS

| Elastic Search | RDBMS |
|:---------------|:------|
| Index          | Database |
| Type | Table |
| Document | Row |
| Field | Column |
| Mapping | Schema |

## CRUD

```bash
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