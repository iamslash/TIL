- [Why do we need URL shortening?](#why-do-we-need-url-shortening)
- [Requirements and Goals of the System](#requirements-and-goals-of-the-system)
  - [Functional Requirements](#functional-requirements)
  - [Non-Functional Requirements](#non-functional-requirements)
  - [Extended Requirements](#extended-requirements)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
  - [traffic estimates](#traffic-estimates)
  - [Storage estimates](#storage-estimates)
  - [Bandwidth estimates](#bandwidth-estimates)
  - [Memory estimates](#memory-estimates)
  - [High-level estimates](#high-level-estimates)
- [System APIs](#system-apis)
- [Database Design](#database-design)
- [Basic System Design and Algorithm](#basic-system-design-and-algorithm)
  - [Encoding actual URL](#encoding-actual-url)
  - [Generating keys offline](#generating-keys-offline)
- [Data Partitioning and Replication](#data-partitioning-and-replication)
- [Cache](#cache)
- [Load Balancer (LB)](#load-balancer-lb)
- [Purging or DB cleanup](#purging-or-db-cleanup)
- [Telemetry](#telemetry)
- [Security and Permissions](#security-and-permissions)

----

# Why do we need URL shortening?

url 의 길이를 줄여주는 서비스

# Requirements and Goals of the System

## Functional Requirements

## Non-Functional Requirements

## Extended Requirements

# Capacity Estimation and Constraints

## traffic estimates

| Number                                                   | Description                 |
| -------------------------------------------------------- | --------------------------- |
| 100 : 1                                                  | read/write ratio            |
| 500M                                                     | write per month             |
| 50B (100 * 500M)                                         | redirection(read) per month |
| 500M / (30 days * 24 hours * 3600 seconds) = ~200 URLs/s | QPS                         |
| 100 * 200 URLs/s = 20K/s                                 | redirection per sec         |

## Storage estimates

## Bandwidth estimates

## Memory estimates

## High-level estimates

| Number  | Description         |
| ------- | ------------------- |
| 200/s   | New URLs            |
| 20K/s   | URL redirections    |
| 100KB/s | Incoming data       |
| 10MB/s  | Outgoing data       |
| 15TB    | Storage for 5 years |
| 170GB   | Memory for cache    |

# System APIs

# Database Design

# Basic System Design and Algorithm

## Encoding actual URL

## Generating keys offline

# Data Partitioning and Replication

# Cache

# Load Balancer (LB)

# Purging or DB cleanup

# Telemetry

# Security and Permissions