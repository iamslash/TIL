- [Twitter Search?](#Twitter-Search)
- [Requirements and Goals of the System](#Requirements-and-Goals-of-the-System)
- [Capacity Estimation and Contraints](#Capacity-Estimation-and-Contraints)
- [System APIs](#System-APIs)
- [High Level Design](#High-Level-Design)
- [Detailed Component Design](#Detailed-Component-Design)
- [Fault Tolerance](#Fault-Tolerance)
- [Cache](#Cache)
- [Load Balancing](#Load-Balancing)
- [Rangking](#Rangking)

-----

# Twitter Search?

트윗을 검색할 수 있다.

# Requirements and Goals of the System

* 1.5 B total users
* 800 M DAU
* 400 M tweets / day
* 300 bytes / tweet
* 500 M searches / day
* the search query will consist of multiple words combined with AND/OR.

# Capacity Estimation and Contraints

* Storage Capacity
  * `400 M * 300 = 120 GB / day`
  * `120 GB / 25 hrs / 3600 sec = 1.38 MB / sec`

# System APIs

```c
search(api_dev_key,
  search_terms,
  maximum_results_to_return,
  sort,
  page_token)

paramters:
returns: (JSON)
  userID, name, tweet text, tweetID, creation time, number of likes
```

# High Level Design

![](img/DesigningTwitterSearchHighLevelDesign.md)

# Detailed Component Design

* Storage
  * `120 GB * 365 days * 5 years = 200 TB`
* How can we create system-wide unique TweetIDs?
  * `400 M * 365 days * 5 years = 730 B`
* Index
* Sharding based on Words
* Sharding based on the tweet object

# Fault Tolerance

# Cache

* In front of Database like [Memcached](https://en.wikipedia.org/wiki/Memcached).
* LRU

# Load Balancing

* Between Clients and Application servers
* Between Application servers and Backend server

# Rangking

* rank tweets by popularity
* The aggregator server combines all these results and sorts them based on the popularity number

