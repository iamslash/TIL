- [Requirements and Goals of the System](#requirements-and-goals-of-the-system)
- [Capacity Estimation and Contraints](#capacity-estimation-and-contraints)
  - [Traffic Estimation](#traffic-estimation)
  - [Storage Estimation](#storage-estimation)
  - [Bandwith Estimation](#bandwith-estimation)
  - [High-level Estimation](#high-level-estimation)
- [System APIs](#system-apis)
- [High Level Architecture](#high-level-architecture)
- [Low Level Architecture](#low-level-architecture)
  - [Database Schema](#database-schema)
  - [TimelineGeneration](#timelinegeneration)
  - [Monitoring](#monitoring)
  - [Extended Requirements](#extended-requirements)
- [System Extentions](#system-extentions)
  - [Data Sharding](#data-sharding)
  - [Cache](#cache)
  - [Replication and Fault Tolerance](#replication-and-fault-tolerance)
  - [Load Balancing](#load-balancing)
- [Q&A](#qa)
- [References](#references)

-----

# Requirements and Goals of the System

* Functional Requirements
  * 유저는 새로운 트윗을 작성할 수 있다.
  * 유저는 다른 유저를 팔로우할 수 있다.
  * 유저는 트윗를 북마크할 수 있다.
  * 서비스는 유저의 타임라인을 생성할 수 있다.
  * 서비스는 유저가 팔로우하는 사람들의 탑 트윗들을 보여줄 수 있다.
  * 트윗은 사진과 동영상도 포함시킬 수 있다.

* Non-functional Requirements
  * highly available
  * 200 ms 안에 타임라인을 생성해야 한다.

* Extended Requirements

  * Searching for tweets.
  * Replying to a tweet.
  * Trending topics – current hot topics/searches.
  * Tagging other users.
  * Tweet Notification.
  * Who to follow? Suggestions?
  * Moments.

# Capacity Estimation and Contraints

## Traffic Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 1 billion   | RU (Registered User) |
| 200 Million   | DAU (Daily Active User) |
| 100 Million   | tweets per day |
| 200 people  | average follow for each user |
| 5 | favorite five tweets for each user per day|
| 1 billion (200 M * 5 favorites)  | total favorite tweets per day|
| 2 times | average visit time lines for each user |
| 20 tweets | user see 20 tweets when they visit their time lines |
| 28 Billion per day (200 M * ((2 + 5) * 20 tweets)) | total tweet-views |

## Storage Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 140 characters  | each tweet    |
| 280 bytes | each tweet    |
| 30 GB / day (100 M * (280 + 30)) | total storage per day |
| 200 KB | photo size per 5 tweets |
| 2 MB | video size per 10 tweets |
| 24 TB / day ((100 M / 5 photos * 200 KB) + ( 100 M / 10 photos * 2 MB)) | media size per day |

## Bandwith Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 24 TB   | ingress per day |
| 290 MB / sec (24 / 86400) | ingress per sec |
| 93 MB / sec (28 Billion * 280 bytes / 86400 sec) | egress text per sec |
| 13 GB / sec (28 Billion / 5 * 280 bytes / 86400 sec) | egress photo per sec | 
| 22 GB / sec (28 Billion / 10 * 280 bytes / 86400 sec) | egress photo per sec | 
| 35 GB / sec | egress per sec |

## High-level Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 24 TB | storage for each day  |
| 43 PB | storage for 5 years  |
| 290 MB / sec (24 / 86400) | ingress per sec |
| 35 GB / sec | egress per sec |

# System APIs

```c
tweet(api_dev_key, 
  tweet_data, 
  tweet_location, 
  user_location,
  media_ids,
  maximum_results_to_return)

parameters:
returns: url to access that tweet  
```

# High Level Architecture

![](DesigningTwitterHighLevelArch)

# Low Level Architecture

## Database Schema

|  | Table |  |
|:-|:------|:-|
|  | **Tweet**  |  |
| PK | TweetID | int | 
| | UserID | int |
| | TweetLatitude | int |
| | TweetLongitude | int |
| | UserLatitude | int |
| | USerLongitude | int |
| | CreationDate | DateTime |
| | NumFavorites | int |
| | | |
|  | **User**  |  |
| PK | UserID | int |
| | Name | varchar(20) |
| | Email | varchar(32) |
| | DateOfBirth | DateTime |
| | CreationDate | DateTime |
| | LastLogin | DateTime |
| | | |
|  | **UserFollow**  |  |
| PK | UserID1 | int |
| PK | UserID2 | int |
| | | |
|  | **Favorite**  |  |
| PK | TweetID | int |
| PK | UserID | int |
| | CreationDate | DateTime |

## TimelineGeneration

[Designing Facebook’s Newsfeed](DesigningFacebooksNewsfeed.md)

## Monitoring
  
* New tweets per day/second, what is the daily peak?
* Timeline delivery stats, how many tweets per day/second our service is delivering.
* Average latency that is seen by the user to refresh timeline.

## Extended Requirements

* How do we serve feeds? 
* Retweet
* Trending Topics
* Who to follow? How to give suggestions? 
* Moments
* Search
  * [Designing Twitter Search](DesigningTwitterSearch.md)

# System Extentions

## Data Sharding

* Sharding based on UserID
* Sharding based on TweetID
* Sharding based on Tweet creation time
* What if we can combine sharding by TweetID and Tweet creation time
  
## Cache

* Which cache replacement policy would best fit our needs?
  * LRU
* How can we have a more intelligent cache? 
  * 80-20 rule, 20 % of tweets generating 80 % of read traffic
* What if we cache the latest data?

## Replication and Fault Tolerance

read replication because of read-heavy and fault tolerance because of replication.

## Load Balancing

* Between Clients and Application Servers
* Between Application servers and database replication servers
* Between Aggregation servers and Cache server

![](img/DesigningTwitterLoadBalancing.png)

# Q&A

# References
