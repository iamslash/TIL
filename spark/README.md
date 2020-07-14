# Abstract



# Materials

* [Quick Start @ Spark](https://spark.apache.org/docs/latest/quick-start.html)
* [[PySpark] Elasticsearch Index 에서 DataFrame 생성하기](https://oboki.net/workspace/python/pyspark-elasticsearch-index-%ec%97%90%ec%84%9c-dataframe-%ec%83%9d%ec%84%b1%ed%95%98%ea%b8%b0/)
* [Spark Framework 에서 Big Data 는 어떻게 분석해야 하는가?](https://niceguy1575.tistory.com/97)

# Install with Docker

* [docker-pyspark @ github](https://github.com/masroorhasan/docker-pyspark)

```console
$ docker pull masroorhasan/pyspark

$ docker run -it --rm masroorhasan/pyspark
> apt-get update
> apt-get install vim curl wget unzip -y
> pyspark
```

Install elasticsearch-hadoop and test.

```console
> cd
> wget https://artifacts.elastic.co/downloads/elasticsearch-hadoop/elasticsearch-hadoop-6.4.1.zip
> unzip elasticsearch-hadoop-6.4.1.zip
> pyspark --driver-class-path=/root/elasticsearch-hadoop-6.4.1/dist/elasticsearch-hadoop-6.4.1.jar

>>> from pyspark.sql import SQLContext
>>> sqlContext = SQLContext(sc)
>>> df = sqlContext.read.format("org.elasticsearch.spark.sql").option("spark.driver.allowMultipleContexts","true").option("es.index.auto.create","true").option("es.nodes.discovery","false").option("es.mapping.id","uuid").option("es.mapping.exclude","uuid").option("es.nodes","xxx.xxx.xxx.xxx").option("es.port","80").option("es.nodes.wan.only","true").load("iamslash/helloworld")
>>> df.registerTempTable("tab")
>>> output = sqlContext.sql("SELECT distinct request FROM tab")
>>> output.show()
```

this is about HTTPS to Elasticsearch.

```py
>>> df = sqlContext.read.format("org.elasticsearch.spark.sql").option("spark.driver.allowMultipleContexts","true").option("es.index.auto.create","true").option("es.nodes.discovery","false").option("es.mapping.id","uuid").option("es.mapping.exclude","uuid").option("es.nodes","xxx.xxx.xxx.xxx").option("es.port","443").option("es.net.ssl","true").option("es.nodes.wan.only","true").load("iamslash/helloworld")
```
