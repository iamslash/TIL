# What is a Rate Limiter?

# Why do we need API rate limiting?

* Misbehaving licents/scripts
* Security
* To prevent abusive behavior and bad design practices
* To keep costs and resource usage under control
* Revenue
* To eliminate spikiness in traffic

# Requirements and Goals of the System

# How to do Rate Limiting?

# What are different types of throttling?

* Hard Throttling
* Soft Throttling
* Elastic or Dynamic Throttling

# What are different types of algorithms used for Rate Limiting?

* Fixed window algorithm
* Rolling window algorithm

# High level design for Rate Limiter

# Basic System Design and Algorithm

# Sliding Window algorithm

# Sliding Window with Counters


| Number                                                                         | Description     |
| ------------------------------------------------------------------------------ | --------------- |
| 8 + (4 + 2 + 20 (Redis hash overhead)) * 60 + 20 (hash-table overhead) = 1.6KB | one user's data |
| 1.6KB * 1 million ~= 1.6GM | |

# Data Sharding and Caching

# Should we rate limit by IP or by user?


