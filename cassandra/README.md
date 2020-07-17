# Materials

* [Cassnadra 의 기본 특징 정리](https://nicewoong.github.io/development/2018/02/11/cassandra-feature/)
* [Apache Cassandra 톺아보기 - 1편](https://meetup.toast.com/posts/58)
* [How To Install Cassandra and Run a Single-Node Cluster on Ubuntu 14.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-install-cassandra-and-run-a-single-node-cluster-on-ubuntu-14-04)

# Install with Docker

* [cassandra @ docker-hub](https://hub.docker.com/_/cassandra)

```console
$ docker run --name my-cassandra -d cassandra

$ docker exec -it my-cassandra bash
> cqlsh
```

# Basics

## Useful Queries 

```console
cqlsh> 
```

## Basic Schema Design

* [Basic Rules of Cassandra Data Modeling](https://www.datastax.com/blog/2015/02/basic-rules-cassandra-data-modeling)
* [Data Modeling in Cassandra @ baeldung](https://www.baeldung.com/cassandra-data-modeling)
