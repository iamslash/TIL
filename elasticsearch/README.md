- [Materials](#materials)
- [ElasticSearch 8.x from 2.x](#elasticsearch-8x-from-2x)
  - [Elasticsearch 2.x vs 8.x 차이점 정리](#elasticsearch-2x-vs-8x-차이점-정리)
  - [주요 변화 상세 설명](#주요-변화-상세-설명)
    - [1. 보안 기본 탑재](#1-보안-기본-탑재)
    - [2. `_type` 제거](#2-_type-제거)
    - [3. Script 엔진: Groovy → Painless](#3-script-엔진-groovy--painless)
    - [4. 역할 기반 노드 구성](#4-역할-기반-노드-구성)
    - [5. KNN, 벡터 검색 지원](#5-knn-벡터-검색-지원)
    - [6. REST API 변화](#6-rest-api-변화)
  - [사용자 관점 주요 주의 사항](#사용자-관점-주요-주의-사항)
  - [요약](#요약)
- [Install](#install)
  - [Install Elasticsearch 2.4.1 from the source](#install-elasticsearch-241-from-the-source)
  - [Install with Docker](#install-with-docker)
  - [Install ElasticSearch 8.18 With Docker Compose](#install-elasticsearch-818-with-docker-compose)
  - [Install Sample Data](#install-sample-data)
- [ElasticSearch Config](#elasticsearch-config)
  - [기본 설정 파일 위치](#기본-설정-파일-위치)
  - [항목별 설정 설명](#항목별-설정-설명)
    - [1. 클러스터 및 노드 설정](#1-클러스터-및-노드-설정)
    - [2. 네트워크 및 포트 설정](#2-네트워크-및-포트-설정)
    - [3. 클러스터 초기 부트스트랩](#3-클러스터-초기-부트스트랩)
    - [4. 데이터 및 로그 저장 경로](#4-데이터-및-로그-저장-경로)
    - [5. 보안 설정 (X-Pack 기본 포함됨)](#5-보안-설정-x-pack-기본-포함됨)
    - [6. 메모리 및 시스템 자원 설정](#6-메모리-및-시스템-자원-설정)
    - [7. 스크립트 및 플러그인 설정](#7-스크립트-및-플러그인-설정)
    - [8. 기타 운영 관련 설정](#8-기타-운영-관련-설정)
  - [실습용 최소 설정 예시 (`elasticsearch.yml`)](#실습용-최소-설정-예시-elasticsearchyml)
  - [운영 환경 설정 예시 (`elasticsearch.yml`)](#운영-환경-설정-예시-elasticsearchyml)
  - [요약](#요약-1)
  - [`discovery.type`](#discoverytype)
    - [`discovery.type`의 주요 후보 값](#discoverytype의-주요-후보-값)
    - [기본 동작: `zen` (Zen Discovery)](#기본-동작-zen-zen-discovery)
    - [특별한 경우: `single-node`](#특별한-경우-single-node)
    - [그 외 클라우드 전용 discovery type 사용 시 주의](#그-외-클라우드-전용-discovery-type-사용-시-주의)
    - [정리](#정리)
  - [`node.roles`](#noderoles)
    - [node.roles 전체 목록 및 상세 설명](#noderoles-전체-목록-및-상세-설명)
    - [주요 조합 예시](#주요-조합-예시)
      - [1. 전체 기능 단일 노드 (개발 환경)](#1-전체-기능-단일-노드-개발-환경)
      - [2. 마스터 전용 노드](#2-마스터-전용-노드)
      - [3. 데이터 전용 노드](#3-데이터-전용-노드)
      - [4. Ingest 전용 노드](#4-ingest-전용-노드)
      - [5. Transform + ML 전용 노드](#5-transform--ml-전용-노드)
      - [6. Hot-Warm-Cool-Archive 티어 구성 예시](#6-hot-warm-cool-archive-티어-구성-예시)
    - [주의사항](#주의사항)
    - [요약](#요약-2)
- [Integration with Spring](#integration-with-spring)
- [ElasticSearch Code Tour](#elasticsearch-code-tour)
- [Basic](#basic)
  - [Elasticsearch 쿼리 전체 종류 (기본적인 분류)](#elasticsearch-쿼리-전체-종류-기본적인-분류)
  - [Elastic \_cat api](#elastic-_cat-api)
  - [Elastic Search vs RDBMS](#elastic-search-vs-rdbms)
  - [CRUD](#crud)
  - [Update](#update)
  - [Bulk](#bulk)
  - [Mapping](#mapping)
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
    - [정의](#정의)
    - [어떻게 동작하나?](#어떻게-동작하나)
    - [예시](#예시)
    - [실무 팁](#실무-팁)
    - [참고 사항](#참고-사항)

----

# Materials

* [ElasticSearch Cheat Sheet 2.x, 7.x](https://elasticsearch-cheatsheet.jolicode.com/#es2)
* [Spring and Elasticsearch @ github](https://github.com/ahnjunwoo/elasticsearchDockerExam)
* [elk @ TIL](/elk/README.md)
* [elasticsearch 실습 정리 @ github](https://github.com/itmare/es)
* [Elastic 가이드 북](https://esbook.kimjmin.net/)

# ElasticSearch 8.x from 2.x

Elasticsearch 2.x와 8.x는 **기능, 아키텍처, 보안, 쿼리 문법, 운영 방식 등 거의 모든 영역에서 큰 변화**가 있었습니다. 아래는 **Elasticsearch 8.18 기준**으로, 2.x와 비교해 항목별로 가장 중요한 변화와 차이점을 정리한 내용입니다.

---

## Elasticsearch 2.x vs 8.x 차이점 정리

| 항목                        | Elasticsearch 2.x         | Elasticsearch 8.x                                                     |
| ------------------------- | ------------------------- | --------------------------------------------------------------------- |
| **Lucene 버전**             | Lucene 5.x                | Lucene 9.x                                                            |
| **Cluster 통신**            | 자체 protocol (TCP)         | 기본은 **HTTPS+TLS**, node 간 통신도 TLS                                     |
| **보안 (X-Pack)**           | 유료 또는 별도 설치 필요            | **기본 내장**, TLS, RBAC, API Key 모두 포함                                   |
| **REST API 구조**           | 단순, 타입 기반                 | **type 제거**, 더 명확하고 JSON 스키마 기반                                       |
| **Mapping 구조**            | 다중 `_type` 지원             | **하나의 인덱스 = 하나의 `_doc` 타입만 허용**                                       |
| **Pipeline / Ingest**     | 존재하지 않음                   | **Ingest Pipeline** 도입 (처리 전 ETL 가능)                                  |
| **Script 엔진**             | Groovy, MVEL (보안 이슈 있음)   | **Painless** 스크립트 도입 (보안 + 성능 개선)                                     |
| **Query DSL**             | 일부 쿼리만 존재 (match, term 등) | `script_score`, `rank_feature`, `dense_vector`, `knn` 등 **강력한 쿼리 지원** |
| **클러스터 노드 역할**            | master, data, client      | **master, data, ingest, ml, transform, voting\_only 등 세분화**           |
| **Data Node**             | 개념 없음, 그냥 data 역할 노드      | **Data Node** 역할 명확히 분리됨 (role flag로 제어)                              |
| **ILM (Index Lifecycle)** | 없음                        | **ILM 도입**: rollover, shrink, delete 등 자동 처리 가능                       |
| **Vector 지원 (Embedding)** | 없음                        | `dense_vector`, `knn`, `ANN` 기반 검색 지원                                 |
| **Ranking plugin**        | custom plugin 필수          | **rank\_feature**, `script_score`, `function_score` 개선                |
| **Kibana 통합**             | 일부 기능만 가능                 | **Fleet, Observability, Security 전면 통합**                              |
| **Snapshot, CCR**         | 수동 구성                     | **Snapshot lifecycle, Cross Cluster Replication 기본 제공**               |
| **Java API**              | TransportClient           | **폐지됨**, 대신 **REST High Level Client**, Java API Client               |
| **정책적 제약**                | 제한 적음 (보안/샌드박스 약함)        | 스크립트, 메모리, 리소스 제한 등 **안정성 강화**                                        |
| **Deprecated API**        | 여전히 다수 사용 가능              | 대부분 제거됨. `_type`, `_timestamp`, `_ttl` 등 사용 불가                        |
| **ML/Monitoring**         | 없음, 외부 필요                 | 기본 내장 (ML anomaly detection, node stat 분석 등)                          |

---

## 주요 변화 상세 설명

### 1. 보안 기본 탑재

* 2.x: `x-pack` 또는 `shield`를 유료로 별도 설치해야 했음
* 8.x: 기본 설치만으로도 TLS, 사용자 인증, 역할 기반 권한(RBAC), Kibana 로그인 보안 모두 제공

### 2. `_type` 제거

* 2.x에서는 하나의 인덱스 안에 여러 타입을 넣는 것이 가능했음
* 8.x에서는 모든 인덱스는 `_doc`이라는 단일 타입만 허용됨

예전:

```json
POST myindex/user/1
```

이제는:

```json
POST myindex/_doc/1
```

### 3. Script 엔진: Groovy → Painless

* Groovy, MVEL 등은 보안 문제로 폐지됨
* `Painless`라는 자체 개발된 스크립트 언어 도입 (속도 빠르고 보안 제약 포함)

### 4. 역할 기반 노드 구성

* 2.x: master/data/client 구분만 존재
* 8.x: `node.roles: [master, data, ingest, ml, transform]` 등으로 세분화

### 5. KNN, 벡터 검색 지원

* 2.x에서는 Embedding 기반 유사도 검색은 외부 시스템 필요 (예: Faiss)
* 8.x는 `dense_vector` 필드, cosine similarity, knn 쿼리 지원

### 6. REST API 변화

* 예전 방식의 쿼리, 인덱스 생성 방식은 대부분 deprecated
* ex: `_type`, `_timestamp`, `_ttl`, `_id`에 의존한 설계 → 제거됨

---

## 사용자 관점 주요 주의 사항

| 구분              | 변화점                                                                                     |
| --------------- | --------------------------------------------------------------------------------------- |
| 인덱스 생성 시        | mapping에서 `_type` 제거 필요                                                                 |
| script 사용 시     | `painless` 외 스크립트 언어는 기본 차단됨                                                            |
| 클러스터 구성 시       | `discovery.zen` → `discovery.seed_hosts`, `cluster.initial_master_nodes`                |
| 보안              | 기본 TLS + 인증 필요 (단일 노드에서도)                                                               |
| Kibana 설정       | `kibana.yml` 보안 연결 (`elasticsearch.username`, `elasticsearch.password`, `ssl` 관련 필드) 필수 |
| TransportClient | 삭제됨 → Java REST Client 또는 Elasticsearch Java API 사용해야 함                                 |

---

## 요약

| 영역      | 큰 변화 유무 | 필요 역량                                    |
| ------- | ------- | ---------------------------------------- |
| 보안      | 매우 큼    | TLS/사용자 관리 숙지 필요                         |
| 쿼리 DSL  | 큼       | `script_score`, `function_score`, 벡터 등   |
| 운영      | 큼       | 노드 역할, ILM, Snapshot 정책                  |
| 플러그인 개발 | 큼       | Java Plugin → Gradle, Descriptor 등 필요    |
| 개발자 도구  | 중간      | Kibana Dev Tools, REST client 방식 익숙해져야 함 |

---




# Install

## Install Elasticsearch 2.4.1 from the source

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

## Install with Docker

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

## Install ElasticSearch 8.18 With Docker Compose

```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.18.0
    container_name: es-david
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - bootstrap.memory_lock=true
      - ES_JAVA_OPTS=-Xms2g -Xmx2g
    ulimits:
      memlock:
        soft: -1
        hard: -1
    ports:
      - "9200:9200"
      - "9300:9300"
    networks:
      - esnet
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  kibana:
    image: docker.elastic.co/kibana/kibana:8.18.0
    container_name: kibana-david
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    networks:
      - esnet
    depends_on:
      elasticsearch:
        condition: service_healthy

networks:
  esnet:
    driver: bridge
```

```bash
$ docker-compose up

# Connect to es-david
$ docker exec -it es-david bash

# Open browser http://localhost:5601
```

## Install Sample Data

* [샘플 데이터 로드](https://www.elastic.co/guide/kr/kibana/current/tutorial-load-dataset.html)
  * 샘플 데이터를 다운로드해서 bulk 로 입력할 수 있다.

```bash
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/bank/account/_bulk?pretty' --data-binary @accounts.json
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/shakespeare/_bulk?pretty' --data-binary @shakespeare.json
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/_bulk?pretty' --data-binary @logs.jsonl
```

# ElasticSearch Config

아래는 Elasticsearch 8.18의 `elasticsearch.yml` 구성 파일에 대한 항목별 설명입니다. 이 설명은 **이해 중심**, **실습/운영 환경 분리 고려**, 그리고 **`david`라는 클러스터명 사용**을 기준으로 작성되었습니다. 

---

## 기본 설정 파일 위치

* Docker 컨테이너: `/usr/share/elasticsearch/config/elasticsearch.yml`
* 리눅스 설치 (tar.gz, rpm): `/etc/elasticsearch/elasticsearch.yml`

---

## 항목별 설정 설명

### 1. 클러스터 및 노드 설정

| 항목             | 설명                      | 예시                                   |
| -------------- | ----------------------- | ------------------------------------ |
| `cluster.name` | 클러스터 이름 (모든 노드가 동일해야 함) | `cluster.name: david-es-cluster`     |
| `node.name`    | 노드의 고유 이름               | `node.name: david-node-1`            |
| `node.roles`   | 노드가 담당할 역할              | `node.roles: [master, data, ingest]` |

> Elasticsearch 8.x부터는 `node.roles`를 명시적으로 지정하는 것이 권장됩니다.

---

### 2. 네트워크 및 포트 설정

| 항목               | 설명                | 예시                                 |
| ---------------- | ----------------- | ---------------------------------- |
| `network.host`   | 노드가 바인딩할 IP 주소    | `network.host: 0.0.0.0` (모든 인터페이스) |
| `http.port`      | REST API가 사용하는 포트 | `http.port: 9200`                  |
| `transport.port` | 클러스터 내부 통신 포트     | `transport.port: 9300`             |

---

### 3. 클러스터 초기 부트스트랩

| 항목                             | 설명                      | 예시                                                       |
| ------------------------------ | ----------------------- | -------------------------------------------------------- |
| `discovery.type`               | 단일 노드로 실행할 경우 설정        | `discovery.type: single-node`                            |
| `discovery.seed_hosts`         | 마스터 노드 후보들의 주소 목록       | `discovery.seed_hosts: ["david-node-1", "david-node-2"]` |
| `cluster.initial_master_nodes` | 클러스터 초기화 시 마스터 후보 노드 목록 | `cluster.initial_master_nodes: ["david-node-1"]`         |

---

### 4. 데이터 및 로그 저장 경로

| 항목          | 설명          | 예시                                  |
| ----------- | ----------- | ----------------------------------- |
| `path.data` | 데이터 저장 디렉토리 | `path.data: /var/lib/elasticsearch` |
| `path.logs` | 로그 저장 디렉토리  | `path.logs: /var/log/elasticsearch` |

---

### 5. 보안 설정 (X-Pack 기본 포함됨)

| 항목                                     | 설명                    | 예시                                           |
| -------------------------------------- | --------------------- | -------------------------------------------- |
| `xpack.security.enabled`               | 인증 및 권한 기능 활성화        | `xpack.security.enabled: true`               |
| `xpack.security.http.ssl.enabled`      | REST API에 TLS 적용      | `xpack.security.http.ssl.enabled: true`      |
| `xpack.security.transport.ssl.enabled` | 노드 간 TLS 적용           | `xpack.security.transport.ssl.enabled: true` |
| `xpack.security.enrollment.enabled`    | 자동 등록 기능 (초기 구성 시 사용) | `xpack.security.enrollment.enabled: true`    |

> 테스트 목적이라면 `xpack.security.enabled: false`로 비활성화 가능하지만, 운영 환경에서는 활성화가 필수입니다.

---

### 6. 메모리 및 시스템 자원 설정

해당 설정은 보통 `jvm.options` 파일 또는 Docker 환경 변수에서 지정합니다.

| 항목                      | 설명                 | 예시                            |
| ----------------------- | ------------------ | ----------------------------- |
| `ES_JAVA_OPTS`          | JVM 힙 메모리 설정       | `-Xms2g -Xmx2g`               |
| `bootstrap.memory_lock` | 메모리 잠금 (GC 안정화 목적) | `bootstrap.memory_lock: true` |

---

### 7. 스크립트 및 플러그인 설정

| 항목                        | 설명           | 예시                                        |
| ------------------------- | ------------ | ----------------------------------------- |
| `script.allowed_types`    | 허용되는 스크립트 타입 | `script.allowed_types: inline, stored`    |
| `script.allowed_contexts` | 허용되는 실행 컨텍스트 | `script.allowed_contexts: search, update` |

---

### 8. 기타 운영 관련 설정

| 항목                                  | 설명                 | 예시                           |
| ----------------------------------- | ------------------ | ---------------------------- |
| `action.auto_create_index`          | 인덱스 자동 생성 허용       | `true` 또는 `"logs-*"`         |
| `cluster.routing.allocation.enable` | 샤드 할당 설정           | `all`, `none`, `primaries` 등 |
| `indices.fielddata.cache.size`      | fielddata 캐시 용량 제한 | `40%`, `512mb` 등             |

---

## 실습용 최소 설정 예시 (`elasticsearch.yml`)

```yaml
cluster.name: david-es-cluster
node.name: david-node-1
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
```

---

## 운영 환경 설정 예시 (`elasticsearch.yml`)

```yaml
cluster.name: david-prod-cluster
node.name: david-node-master
node.roles: [master]
network.host: 192.168.1.100
discovery.seed_hosts: ["192.168.1.100", "192.168.1.101"]
cluster.initial_master_nodes: ["david-node-master", "david-node-1"]

xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.certificate: certs/node.crt
xpack.security.transport.ssl.key: certs/node.key
xpack.security.transport.ssl.certificate_authorities: ["certs/ca.crt"]
```

---

## 요약

| 범주      | 주요 항목                                                                    |
| ------- | ------------------------------------------------------------------------ |
| 클러스터 구성 | `cluster.name`, `node.name`, `node.roles`                                |
| 네트워크    | `network.host`, `http.port`, `transport.port`                            |
| 보안      | `xpack.security.*`                                                       |
| 부트스트랩   | `discovery.type`, `discovery.seed_hosts`, `cluster.initial_master_nodes` |
| 경로      | `path.data`, `path.logs`                                                 |
| 성능 안정성  | `bootstrap.memory_lock`, `ES_JAVA_OPTS`                                  |

## `discovery.type`

`discovery.type`은 Elasticsearch 클러스터가 **마스터 노드를 어떻게 탐색(discovery)** 할지를 결정하는 설정입니다.
`discovery.type`의 값은 운영 방식에 따라 달라지며, **`single-node` 외에도 몇 가지 특별한 값이 존재합니다.**

---

### `discovery.type`의 주요 후보 값

| 값                          | 설명                                               |
| -------------------------- | ------------------------------------------------ |
| `single-node`              | 단일 노드로 실행 (마스터 선출 과정 없음). **개발/테스트 전용**          |
| `zen` (기본값)                | 다중 노드 클러스터에서 사용하는 표준 방식 (Zen Discovery)          |
| `file`                     | 정적인 파일 기반 노드 목록을 사용하여 마스터 탐색                     |
| `ec2` (플러그인)               | AWS EC2 환경에서 인스턴스를 자동 탐색 (Elastic AWS plugin 필요) |
| `gce` (플러그인)               | Google Cloud Platform(GCE) 기반 인스턴스 탐색            |
| `azure` (플러그인)             | Azure VM 기반 탐색                                   |
| `docker` (내부 전용)           | Docker 환경에서 특수한 discovery 테스트용                   |
| `kubernetes` (비공식 플러그인 기반) | Kubernetes 환경에서 Pod 기반 discovery 사용              |
| `noop` (테스트 전용)            | 마스터 탐색을 하지 않음. **테스트 전용 내부 구현**                  |

---

### 기본 동작: `zen` (Zen Discovery)

`discovery.type`을 생략하거나 명시하지 않으면 기본은 `zen`입니다.

Zen Discovery는 다음 설정들과 함께 사용됩니다:

```yaml
discovery.seed_hosts: ["es-master-1", "es-master-2"]
cluster.initial_master_nodes: ["es-master-1"]
```

이 설정들은 노드들이 서로를 찾아 마스터를 선출하는 데 사용됩니다.

---

### 특별한 경우: `single-node`

```yaml
discovery.type: single-node
```

* 단일 노드에서 실행 시 편의상 마스터 선출과정 생략
* `cluster.initial_master_nodes` 불필요
* `discovery.seed_hosts`도 생략 가능
* Kibana와 연동, 플러그인 테스트, 로컬 실습에 적합

---

### 그 외 클라우드 전용 discovery type 사용 시 주의

* `ec2`, `gce`, `azure` 등은 기본 Elasticsearch에 포함되어 있지 않으며, **Elastic Cloud plugin 또는 Elastic Cloud Enterprise 환경**에서 사용됩니다.
* 직접 쓸 경우 해당 plugin을 수동으로 설치해야 합니다:

```bash
elasticsearch-plugin install discovery-ec2
```

---

### 정리

| 타입                    | 용도                          |
| --------------------- | --------------------------- |
| `single-node`         | 개발, 단일 노드 테스트               |
| `zen` (기본)            | 모든 운영용 클러스터                 |
| `ec2`, `gce`, `azure` | 클라우드 기반 자동 탐색 (플러그인 필요)     |
| `file`                | 수동 IP 목록 기반 클러스터 구성 (매우 드묾) |
| `noop`, `docker`      | 테스트용, 내부 전용                 |

---

운영 환경에서는 반드시 `zen` 디스커버리와 함께 `discovery.seed_hosts`, `cluster.initial_master_nodes`를 설정해야 합니다.
필요하시면 `zen` 기반 클러스터 구성을 위한 `elasticsearch.yml` 예제도 제공해 드릴 수 있습니다.

## `node.roles`

`node.roles`는 Elasticsearch 7.x 이후부터 도입된 설정으로, 각 노드가 **어떤 역할을 수행할지 명확히 지정**할 수 있도록 만들어졌습니다. Elasticsearch 8.x에서는 기본값이 없으며, **운영 시 `node.roles`를 명시적으로 설정하는 것이 매우 권장**됩니다.

아래는 Elasticsearch 8.18 기준으로 설정 가능한 모든 `node.roles` 옵션들과 그 역할, 특징, 용도를 정리한 표입니다.

---

### node.roles 전체 목록 및 상세 설명

| 역할 이름                   | 설명                                              | 주 용도                       | 관련 기능/제약                        |
| ----------------------- | ----------------------------------------------- | -------------------------- | ------------------------------- |
| `master`                | 클러스터 마스터 자격을 가짐 (선출 가능)                         | 클러스터 상태 관리, 샤드 할당, 설정 변경 등 | 최소 3개 이상 구성 권장                  |
| `data`                  | 데이터를 저장하고 검색/색인 처리 수행                           | 문서 색인, 검색, aggregation     | 가장 일반적인 노드                      |
| `ingest`                | Ingest Pipeline 처리 수행 (예: 필드 파싱, 변환)            | 데이터 전처리, 파이프라인 실행          | ingest pipeline 사용 시 필수         |
| `ml`                    | Elastic ML 기능 (anomaly detection 등) 실행          | Elastic Stack ML 기능 사용 시   | X-Pack 필요                       |
| `transform`             | Transform 작업 수행 (pivot, rollup 등)               | 집계 기반 구조 변경                | Kibana의 Transform UI와 연계        |
| `remote_cluster_client` | 다른 클러스터와 통신 (Cross Cluster Search, Replication) | 다중 클러스터 환경에서 연동            | 기본적으로 모든 노드가 지원, 필요 시 제외 가능     |
| `voting_only`           | 마스터 선출 투표에는 참여하지만 리더가 될 수 없음                    | 고가용성 마스터 투표 참여 전용          | `master`와 함께 사용, 하지만 리더 제외됨     |
| `data_content`          | content 인덱스 전용 데이터 노드                           | 일반적인 문서 저장                 | `data` 역할보다 세분화                 |
| `data_hot`              | hot tier 인덱스 저장용                                | 빠른 읽기/쓰기, 최근 데이터           | ILM의 hot phase 용                |
| `data_warm`             | warm tier 인덱스 저장용                               | 자주 접근하지 않지만 유지해야 하는 데이터    | ILM의 warm phase 용               |
| `data_cold`             | cold tier 인덱스 저장용                               | 거의 접근하지 않지만 삭제하지 않는 데이터    | ILM의 cold phase 용               |
| `data_frozen`           | frozen tier 인덱스 저장용                             | 자주 조회되지 않는 장기 보관용 데이터      | query 시 디스크에서 바로 읽음             |
| `coordinating_only`     | 검색 요청 분산, 집계 연산 중간 처리                           | 검색 전용 로드 밸런서 역할            | `node.roles: []` 또는 roles 모두 제거 |

---

### 주요 조합 예시

#### 1. 전체 기능 단일 노드 (개발 환경)

```yaml
node.roles: [master, data, ingest]
```

#### 2. 마스터 전용 노드

```yaml
node.roles: [master]
```

#### 3. 데이터 전용 노드

```yaml
node.roles: [data_content]
```

#### 4. Ingest 전용 노드

```yaml
node.roles: [ingest]
```

#### 5. Transform + ML 전용 노드

```yaml
node.roles: [ml, transform]
```

#### 6. Hot-Warm-Cool-Archive 티어 구성 예시

| 노드 타입     | 설정                               |
| --------- | -------------------------------- |
| hot 노드    | `node.roles: [data_hot, ingest]` |
| warm 노드   | `node.roles: [data_warm]`        |
| cold 노드   | `node.roles: [data_cold]`        |
| frozen 노드 | `node.roles: [data_frozen]`      |

---

### 주의사항

* `node.roles`는 생략할 수 있으나, 명시하는 것이 클러스터 설계에 도움이 됩니다.
* 일부 역할(`data_hot`, `data_warm`, ...)은 ILM(Index Lifecycle Management) 또는 tiered storage 전략과 밀접하게 연결됩니다.
* `coordinating_only` 노드는 REST 요청을 받아 라우팅하거나, 집계 연산을 처리하지만 데이터나 메타데이터를 저장하지 않습니다.

  ```yaml
  node.roles: []
  ```

---

### 요약

| 분류      | 역할                                                                          |
| ------- | --------------------------------------------------------------------------- |
| 클러스터 관리 | `master`, `voting_only`                                                     |
| 데이터 저장  | `data`, `data_hot`, `data_warm`, `data_cold`, `data_frozen`, `data_content` |
| 전처리     | `ingest`, `transform`                                                       |
| 검색 최적화  | `coordinating_only`                                                         |
| 분석      | `ml`                                                                        |
| 클러스터 연동 | `remote_cluster_client`                                                     |


# Integration with Spring

* [spring-examples/exelasticsearch @ github](https://github.com/iamslash/spring-examples/blob/master/exelasticsearch/README.md)

# ElasticSearch Code Tour

[ElasticSearch Code Tour](es-codetour.md)

# Basic

## Elasticsearch 쿼리 전체 종류 (기본적인 분류)

[ElasticSearch Query](es-query.md)

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

### 정의

`"refresh_interval": "1s"`

이 설정은 Elasticsearch가 세그먼트를 디스크에 반영하고 검색 가능 상태로 전환하는 시간 간격을 의미합니다.

기본값은 "1s" (1초마다 refresh가 수행됨)

### 어떻게 동작하나?

문서를 색인 (index)하면, 그 문서는 바로 검색 가능한 것은 아님 refresh_interval이 지나고 나면, Lucene 세그먼트가 flush되고, 문서가 검색 대상에 포함됨.

### 예시

`refresh_interval: 1s`

→ 문서를 색인하고 약 1초 후 검색 가능

`refresh_interval: 30s`

→ 색인 후 30초가 지나야 검색 가능

`refresh_interval: -1`

→ 자동 refresh를 끄고, 수동으로 `POST /<index>/_refresh` 요청해야만 검색 가능

### 실무 팁

| 상황	| 추천 설정 |
|--|--|
| 실시간 검색이 필요한 경우 (예: 채팅, 검색엔진) |	`1s` (기본값 유지) |
|대량 색인 작업 중 (성능 최적화)	| `"refresh_interval": "-1"` 후 일괄 색인 후 `_refresh` 호출 |
| 색인 지연이 허용되는 경우 (예: 로그 저장)	| `30s` 이상으로 설정 가능 |

### 참고 사항

refresh는 리소스를 사용하는 작업입니다 (디스크/메모리).

너무 짧게 설정하면 성능에 영향을 줄 수 있고,
너무 길게 설정하면 검색 지연이 발생할 수 있습니다.
