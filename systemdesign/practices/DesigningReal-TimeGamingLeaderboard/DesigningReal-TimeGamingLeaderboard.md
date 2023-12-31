- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
  - [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [API Design](#api-design)
  - [High-Level Architecture](#high-level-architecture)
  - [Data Model](#data-model)
  - [Storage Estimation](#storage-estimation)
- [High Level Design Deep Dive](#high-level-design-deep-dive)
  - [Scale Redis](#scale-redis)
- [Extension](#extension)
  - [Faster Retrieval and Breaking Tie](#faster-retrieval-and-breaking-tie)
  - [System Failure Recovery](#system-failure-recovery)
- [Q\&A](#qa)
  - [How To Deal with 5B users](#how-to-deal-with-5b-users)
- [References](#references)

----

# Requirements

## Functional Requirement

* The system displays 10 players on the leaderboard.
* The system shows a user's specific rank.
* The system displays users above and below the specific user.

## Non-Functional Requirement

* The system updates scores in real time.
* The system should be scalable, available, reliable.

## Estimation

| Number | Description| |
|--|--|--|
| 5 Millian | DAU | |
| 50 | Updated users per second | 5,000,000 DAU / 100,000 sec |
| 250 | Peak updated users per second | 50 * 5 |
| 10 | A user plays 10 games per day on average | |
| 500 | QPS for updating score | 50 * 10 |
| 2,500 | Peak QPS for updating score | 500 QPS * 5 |
| 1 | The number of fetching top 10 scores in a day | |
| 50 | QPS for fetching top 10 score | 5,000,000 DAU / 100,000 sec | 

# High Level Design

## API Design

```json
* Update a user's score.
  * POST /v1/scores 
  * Request
    * user_id: The user who wins a game.
    * points: The user's point
  * Response
    * 200 OK: Succeeded to update a user's score.
    * 400 Bad Request: Failed to update a user's score.

* Get top 10 scores.
  * GET /v1/scores
  * Response
  {
    "data": [
    {
      "user_id": "1",
      "user_name": "foo",
      "rank": 1,
      "score": 976
    },
    {
      "user_id": "2",
      "user_name": "bar",
      "rank": 2,
      "score": 966
    },
    ],
    ...
    "total": 10
  }

* Get the rank
  * GET /v1/scores/{:user_id}
  * Request
    * user_id
  * Response
  {
    "user_info": {
      "user_id": "user5",
      "score": 940,
      "rank": 6,
    }
  }
```

## High-Level Architecture

## Data Model

We can think RDBMS, Redis, NoSQL for storages. We will choose single instance of Redis. Redis supports sorted sets.

sorted set is implemented using [skip list](https://github.com/iamslash/learntocode/blob/master/fundamentals/list/skiplist/README.md).

These are operations of sorted sets.

* `ZADD`
* `ZINCRBY`
* `ZRANGE/ZREVRANGE`
* `ZRANK/ZREVRANK`

These are examples of sorted sets.

```
* Update a user's score

ZINCRBY <key> <increment> <user>
ZINCRBY leaderboard_feb_2022 1 'iamslash'

* Get top 10 scores.

ZREVRANGE leaderboard_feb_2022 0 9 WITHSCORES
[(user2,score2),(user1,score1,(user5,score5),...)]

* Get the rank

ZREVRANK leaderboard_feb_2022 'iamslash'

* Get above and below the specific user

iamslash's rank is 361
ZREVRANGE leaderboard_feb_2022 357 365
```

## Storage Estimation

| Number | Description | |
|--|--|--|
| 25 Millian | The number of users which won at least once | |
| 24 bytes | The size of user id | |
| 2 bytes | The size of score | | 
| 650 MB | 25 Millian * 26 bytes | | 

Single Redis can handle `650 MB` for updating in real time and can handle `2,500 peak QPS`.

# High Level Design Deep Dive

## Scale Redis

Scaling up is ok Scaling out makes the system complicated.

# Extension

## Faster Retrieval and Breaking Tie

Cache user information in Redis.

When two users' score are same old one's score is higher than new one's score.

## System Failure Recovery

We can recover Redis with restoring data from RDBMS.

# Q&A

## How To Deal with 5B users

매우 많은 수의 user 가 있다면 어떻게 해결할 수 있을까?

`5 Billian users` 를 위해 `5 Billian * 10 Byte = 50 GB` 가 필요하다. Redis 의
sorted set 은 sharding 이 필요하다. 다음과 같은 2 가지 방법을 생각할 수 있다.

첫째, shard key 를 user 의 순위로 하자. 즉, 순위를 구간별로 나누고 한 구간을
하나의 shard 에 배정한다. top 10 users 를 쉽게 가져올 수 있다. 그러나 특정 user
의 순위를 쉽게가져 올 수는 없다. user 의 순위 구간이 변경되면 전체 user 의
순위가 조정되야 한다. system 의 complexity 가 높다.

두번째, shard key 를 user 의 id 로 하자. 즉, 각 shard 마다 독자적인 순위를
갖는다. client 가 top 10 users 의 정보를 얻으려면 각 shard 의 top 10 users 를
모두 끌어모아서 정렬하고 다시 top 10 users 를 추출한다. 특정 user 의 순위 역시
알아내기가 쉽지 않다. system 의 complexity 가 높다. client 는 필요한 정보보다
shard count 배 만큼 정보를 조회해야 한다. 불필요하다. 

sorted set 을 sharding 하는 것은 system 의 complexity 를 높인다.

다음과 같은 해결책을 생각해 볼 수 있다.

rank 에 관심있는 user 들은 상위권 user 들이다. 상위권 user 들만 Single Instance
Redis 의 sorted set 에 저장한다. top 10 users 는 쉽게 가져올 수 있다.

하위권 user 들은 어림잡아 계산한다. 정확하지 않아도 된다. 예를 들어 점수를
구간별로 나누고 구간에 얼만큼 user 들이 존재하는지 하루에 한번 계산해 Single
Instance Redis 에 저장해 둔다. 특정 user 의 점수를 보고 상위 구간에 얼만큼 user
들이 있는지 알 수 있다.

# References

- [Leaderboard System Design](https://systemdesign.one/leaderboard-system-design/)
