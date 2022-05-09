- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-functional Requirements](#non-functional-requirements)
- [High Level Design](#high-level-design)
  - [High Level Architecture](#high-level-architecture)
  - [System APIs](#system-apis)
- [Low Level Design](#low-level-design)
  - [UUID](#uuid)
  - [Snow Flake](#snow-flake)

----

# Requirements

## Functional Requirements

* ID should be unique.
* ID should be numeric.
* ID should be 64-bit.
* ID should be ordered by date.
* QPS should be over 1000.

## Non-functional Requirements

* High available, reliable

# High Level Design

## High Level Architecture

There are application servers which are synchronized with the
[NTP](https://en.wikipedia.org/wiki/Network_Time_Protocol) server.

## System APIs

```
GET /v1/ids
```

# Low Level Design

## UUID

[UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier) is 128-bit.
This can not be a good solution because the size is too big.

## Snow Flake

twitter invented [snow flake](https://github.com/twitter-archive/snowflake/releases/tag/snowflake-2010) for unique id generator. This can be a good solution.

```
 1 bit (for future)
41 bit timestamp         2^41 = 2199023255552 covers 61 years 
 5 bit datacenter ID      2^5 = 32            covers 32 datacenters
 5 bit machine ID         2^5 = 32            covers 32 machines 
12 bit seq number        2^12 = 4096          covers 4096 numbers at the same millisecond
```
