- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-functional Requirements](#non-functional-requirements)
  - [Extended Requirements](#extended-requirements)
- [System APIs](#system-apis)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
  - [Traffic Estimation](#traffic-estimation)
  - [Storage Estimation](#storage-estimation)
  - [Bandwidth Estimation](#bandwidth-estimation)
  - [High level Estimation](#high-level-estimation)
- [High-level Architecture](#high-level-architecture)
- [Low-level Architecture](#low-level-architecture)
  - [Components](#components)
  - [Schema](#schema)
    - [User Database](#user-database)
    - [Meta Database](#meta-database)
  - [Video, Thumbnail Storage](#video-thumbnail-storage)
  - [Video Deduplication](#video-deduplication)
  - [Load Balancing](#load-balancing)
  - [Cache](#cache)
  - [DataBase Sharding](#database-sharding)
  - [CDN for Video, Thumbnail](#cdn-for-video-thumbnail)
- [System Extentions](#system-extentions)
- [Q&A](#qa)
- [References](#references)

-----

# Requirements


## Functional Requirements

* Users can upload videos.
* Users can share and view videos.
* Users can search based on video titles.
* Users can add, view comments of videos.
* The system can record stats of videos, including likes, dislikes, views, etc.

## Non-functional Requirements

* The system is highly reliable and can't loose nay video.
* The system is highly available and consistency can take a hit.
* Users can watch video with real time, without any lag.

## Extended Requirements

# System APIs

```
uploadVideo(api_key, 
            video_title, 
            video_description,
            tags[], 
            category_id, 
            default_language, 
            recording_details, 
            video)

video(stream) : binary of video

searchVideo(api_key,
            search_query,
            user_location, 
            page_size, 
            page_no,
            page_token)

streamVideo(api_key,
            video_id,
            video_offset,
            codec, 
            resolution)            
```

# Capacity Estimation and Constraints

## Traffic Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 1.5 billion   | RU (Registered User) |
| 800 Million   | DAU (Daily Active User) |
| 5 videos | average view per day for one user |
| 46 K videos/sec (800 million * 5) | view per sec for DAU |
| 1 : 200 | write to read ratio |
| 230 views / sec (46 K / 200) | views per sec for DAU |

## Storage Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 500 hours / min    | 500 hours videos are uploaded per min |
| 50 MB | video size for one min video |
| 25 GB / sec (1500 GB/min = 500 hours * 60 min * 50 MB) | upload video size without compression, replication |

## Bandwidth Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 5 GB / sec (300 GB / min = 500 hours * 60 mins * 10MB)    | ingress per sec |
| 1 TB / sec (5 GB * 200) | egress per sec |

## High level Estimation

| Number          | Description                       |
| --------------- | --------------------------------- |
| 432 TB (5GB * 86400) | storage for each day  |
| 788 PB | storage for 5 years  |

# High-level Architecture

# Low-level Architecture

## Components

* Video Queue
* Encoder
* Thumnal generator
* Video and Thumbnail Storage
* User Database
* Meta Database

## Schema

### User Database

```sql
CREATE TABLE user {
  user_id int,
  name varchar(255), 
  email varchar(255), 
  address varchar(255), 
  age int
};
```

### Meta Database 

```sql
CREATE TABLE video {
  video_id int,
  title varchar(255), 
  description varchar(255), 
  size int, 
  thumbnail var(255),
  uploader var(255),
  likes int,
  dislikes int,
  views int
};

CREATE TABLE comment {
  commment_id int,
  video_id int,
  user_id int,
  comment var(255),
  created_at datetime,
  created_by var(255),
  updated_at datetime,
  updated_by var(255)
};
```

## Video, Thumbnail Storage

HDFS, GluterFS

## Video Deduplication

Chunks

## Load Balancing

## Cache

## DataBase Sharding

## CDN for Video, Thumbnail

# System Extentions

# Q&A

# References

* [Design Video Sharing Service System like Youtube](https://www.geeksforgeeks.org/design-video-sharing-system-like-youtube/)
* [System Design Interview: Mini YouTube](https://medium.com/@eileen.code4fun/system-design-interview-mini-youtube-5cae5eedceae)
* [Seattle Conference on Scalability: YouTube Scalability @ youtube](https://www.youtube.com/watch?v=w5WVu624fY8)
* [GlusterFS basic](https://gruuuuu.github.io/linux/glusterfs/#)
