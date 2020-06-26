- [Abstract](#abstract)
- [Materials](#materials)
- [Install with docker](#install-with-docker)
- [Basic Elastic Search](#basic-elastic-search)
- [Basic Kibana](#basic-kibana)
- [Basic Logstash](#basic-logstash)

----

# Abstract

OLAP (Online Analytical Processing) soution 중 하나인 ELK 에 대해 정리한다.

# Materials

* [ELK 스택 (ElasticSearch, Logstash, Kibana) 으로 데이터 분석 @ inflearn](https://www.inflearn.com/course/elk-%EC%8A%A4%ED%83%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D/lecture/5506)
  * 킹왕짱
  * [src](https://github.com/minsuk-heo/BigData)
* [Elasticsearch, Logstash, Kibana (ELK) Docker image documentation](https://elk-docker.readthedocs.io/)
  * elk docker image
  * [image](https://hub.docker.com/r/sebp/elk)
* [Elastic Stack (6.2.4) 을 활용한 Dashboard 만들기 Project](https://github.com/higee/elastic)

# Install with docker

* [Elasticsearch, Logstash, Kibana (ELK) Docker image](https://hub.docker.com/r/sebp/elk) docker image 를 설치한다.

```console
$ docker search elk
$ docker pull sebp/elk
$ docker run -p 5601:5601 -p 9200:9200 -p 5000:5000 -it --name my-elk sebp/elk
$ docker exec -it my-elk /bin/bash
```

* browser 를 이용하여 `localhost:5601` (kibana) 에 접속한다.
* browser 를 이용하여 `localhost:9200` (elasticsearch) 에 접속한다.

# Basic Elastic Search

* [Elasticsearch @ TIL](/elasticsearch/README.md)

# Basic Kibana

* [Kibana 사용자 가이드](https://www.elastic.co/guide/kr/kibana/current/index.html)
  * 킹왕짱

# Basic Logstash

* [kafka-elk-docker-compose](https://github.com/sermilrod/kafka-elk-docker-compose)
  * dockercompose 로 filebeat, kafka, zookeeper, elk 를 실행한다. 특히 logstash 설정을 참고할 만 함.

logstash 는 input, filter, output plugin 을 설정할 수 있다. 예를 들어 logstash 가 kafka 를 data source 로 한다면 input 항목에 kafka 를 설정한다. 그리고 logstash 가 elasticsearch 를 data destination 으로 한다면 output 항목에 elasticsearch 를 설정한다.

```yml
input {
  kafka {
    bootstrap_servers => "kafka1:9092,kafka2:9092,kafka3:9092"
    client_id => "logstash"
    group_id => "logstash"
    consumer_threads => 3
    topics => ["log"]
    codec => "json"
    tags => ["log", "kafka_source"]
    type => "log"
  }
}

filter {
  if [type] == "apache_access" {
    grok {
      match => { "message" => "%{COMMONAPACHELOG}" }
    }
    date {
      match => ["timestamp", "dd/MMM/yyyy:HH:mm:ss Z"]
      remove_field => ["timestamp"]
    }
  }
  if [type] == "apache_error" {
    grok {
      match => { "message" => "%{COMMONAPACHELOG}" }
    }
    date {
      match => ["timestamp", "dd/MMM/yyyy:HH:mm:ss Z"]
      remove_field => ["timestamp"]
    }
  }
}

output {
  if [type] == "apache_access" {
    elasticsearch {
         hosts => ["elasticsearch:9200"]
         index => "logstash-apache-access-%{+YYYY.MM.dd}"
    }
  }
  if [type] == "apache_error" {
    elasticsearch {
         hosts => ["elasticsearch:9200"]
         index => "logstash-apache-error-%{+YYYY.MM.dd}"
    }
  }
}
```
