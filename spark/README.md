- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Install with Docker](#install-with-docker)
  - [Simple Example Word Count](#simple-example-word-count)

----

# Abstract

**Apache Spark** is an open-source, distributed computing system designed for fast
and flexible big data processing and analytics. It was developed in response to
the limitations of the Hadoop MapReduce computing model, which can be slow and
inflexible for some use cases. Spark provides an in-memory data processing
engine that significantly improves performance for iterative algorithms and
interactive querying.

Some key features of **Apache Spark** include:

- **Speed**: Spark's in-memory processing capabilities enhance the performance
  of data processing tasks, making it much faster than traditional disk-based
  systems like Hadoop MapReduce.
- **Flexibility**: Spark supports multiple programming languages, including
  **Scala**, **Python**, **Java**, and **R**, making it accessible to a wide
  range of developers and data scientists.
- **Ease of use**: Spark provides high-level APIs that simplify complex data
  processing tasks, enabling users to easily develop applications and perform
  analytics.
- **Fault Tolerance**: Spark includes built-in fault-tolerance features through
  data replication and lineage information, ensuring that the system and data
  processing can recover from failures.
- **Scalability**: Spark can scale from a single node to thousands of nodes,
  making it suitable for processing large volumes of data across clusters of
  computers.

In addition to data processing and analytics, Apache Spark also supports
advanced capabilities such as machine learning through MLlib, graph processing
through GraphX, and streaming analytics through Spark Streaming. This makes
Spark a versatile and powerful platform for processing and analyzing massive
amounts of data in real-time.

# Materials

* [Spark 의 핵심은 무엇인가? RDD! (RDD paper review) @ slideshare](https://www.slideshare.net/yongho/rdd-paper-review)
  * Spark 의 핵심을 잘 정리함.
* [Quick Start @ Spark](https://spark.apache.org/docs/latest/quick-start.html)
* [[PySpark] Elasticsearch Index 에서 DataFrame 생성하기](https://oboki.net/workspace/python/pyspark-elasticsearch-index-%ec%97%90%ec%84%9c-dataframe-%ec%83%9d%ec%84%b1%ed%95%98%ea%b8%b0/)
* [Spark Framework 에서 Big Data 는 어떻게 분석해야 하는가?](https://niceguy1575.tistory.com/97)

# Basic

## Install with Docker

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

## Simple Example Word Count

Here's a simple example of using Apache Spark with Python (PySpark) to count the
number of occurrences of each word in a given text file:

```py
from pyspark import SparkConf, SparkContext

# Configure and initialize Spark
conf = SparkConf().setAppName("WordCountApp")
sc = SparkContext(conf=conf)

# Load data from a text file
data = sc.textFile("path/to/your/textfile.txt")

# Split each line into words
words = data.flatMap(lambda line: line.split(" "))

# Map each word to a key-value pair where key=word and value=1
word_counts = words.map(lambda word: (word, 1))

# Reduce the key-value pairs by key (word), adding up the values (counts)
word_counts = word_counts.reduceByKey(lambda a, b: a + b)

# Collect the results and print them
for word, count in word_counts.collect():
    print("{}: {}".format(word, count))

# Stop the SparkContext
sc.stop()
```

Make sure to replace "path/to/your/textfile.txt" with the actual path to the
text file you want to analyze. This simple example demonstrates how to use
Spark's `textFile`, `flatMap`, `map`, and `reduceByKey` transformations to
process a text file and count the occurrences of each word.

Remember that you'll need to have PySpark installed and properly configured for
this example to run. You can install PySpark using pip:

```bash
$ pip install pyspark
```
