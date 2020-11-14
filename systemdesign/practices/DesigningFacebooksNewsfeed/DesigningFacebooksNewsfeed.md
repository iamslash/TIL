- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-functional Requirements](#non-functional-requirements)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
  - [Traffic Estimates](#traffic-estimates)
  - [Storage Estimates](#storage-estimates)
- [System APIs](#system-apis)
- [High-level Architecture](#high-level-architecture)
- [Low-level Architecture](#low-level-architecture)
- [System Extentions](#system-extentions)
- [Q&A](#qa)
- [Implementation](#implementation)
- [References](#references)

-----

# Requirements

## Functional Requirements

* Facebook newsfeed is generated based on the posts from the people, pages, and groups.
* Users follow other users, pages, groups.
* Feeds contain images, videos, text.
* The system append new posts to alive user's feed as they arrive.

## Non-functional Requirements

* The system generate uers' newsfeed in realtime and maximum latency could be 2 sec.

# Capacity Estimation and Constraints

## Traffic Estimates

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 300   | average follows of users |
| 200 | average follows of pages |
| 300 M | DAU |
| 5 times | average checking times of a user |
| 1.5 B (300 M * 5) | Total checking times of users in a day |
| 17,500 (1.5 B / 86400) | aprozimate requests per day |

## Storage Estimates

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 500 posts   | average posts of a user to check |
| 1 KB | average size of a post |
| 500 KB (500 posts * 1 KB) | average post size of a user |
| 150 TB (500 KB * 300 M) | average posts size of users in a day |
| 1500 machines (150 TB / 100 GB) | number of machines for storing posts |

# System APIs

```
get_user_feed(
  api_key,
  user_id,
  since,
  per_page,
  page,
  max_id,
  exclude_replies
  )

since: from date time

```

# High-level Architecture

# Low-level Architecture

# System Extentions

# Q&A

# Implementation

* [fb-clone @ github](https://github.com/rOluochKe/fb-clone)

# References

* [System Design Mock Interview (Dropbox Software Engineer): Design Facebook Newsfeed @ youtube](https://www.youtube.com/watch?v=PDWD6IqU_nQ)
* [Designing Instagram: System Design of News Feed @ youtube](https://www.youtube.com/watch?v=QmX2NPkJTKg)
* [Twitter system design | twitter Software architecture | twitter interview questions @ youtube](https://www.youtube.com/watch?v=wYk0xPP_P_8)
