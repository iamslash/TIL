- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
- [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [API Design](#api-design)
  - [Data Model](#data-model)
    - [Read/Write Ratio](#readwrite-ratio)
    - [Data Schema](#data-schema)
  - [System Architecture Diagram](#system-architecture-diagram)
    - [Load Balancer](#load-balancer)
    - [LBS (Location Based Service)](#lbs-location-based-service)
    - [Business Service](#business-service)
    - [Database Cluster](#database-cluster)
    - [Scalability of Business Service and LBS](#scalability-of-business-service-and-lbs)
  - [Algorithms To Find Nearby Businesses](#algorithms-to-find-nearby-businesses)
    - [Two-Dimensional Search](#two-dimensional-search)
    - [Evenly Divided Grid](#evenly-divided-grid)
    - [Geohash](#geohash)
    - [Quadtree](#quadtree)
    - [Google S2](#google-s2)
  - [Geohash vs Quadtree](#geohash-vs-quadtree)
- [High Level Deisgn Deep Dive](#high-level-deisgn-deep-dive)
  - [Scale The Database](#scale-the-database)
  - [Caching](#caching)
  - [Region and Availability Zones](#region-and-availability-zones)
  - [Filter Results by Time or Business Type](#filter-results-by-time-or-business-type)
  - [Final Architecture Diagram](#final-architecture-diagram)

----

# Requirements

## Functional Requirement

* Return all shops on a user's location (latitude and longitude pair) and radius.
* Shop owners can add, delete or update a shop, but this information does not need to be reflected in real-time.
* Customers can view detailed information of shops.

## Non-Functional Requirement

* Low latency.
* Data privacy including GDPR (General Data Protection Regulation) and CCPA (California Consumer Privacy Act), etc.
* High availability and scalability.

# Estimation

| Number | Description | Calculation |
|--|--|--|
| 100 Millian | DAU | |
| 86,400 sec (=~ 100,000 sec) | seconds ina day | |
| 5,000 | Search QPS | 100,000,000 DAU / 100,000 sec | 

# High Level Design

## API Design

> * [Place Search | Google](https://developers.google.com/maps/documentation/places/web-service/search)
> * [Search Businesses | Yelp](https://docs.developer.yelp.com/reference/v3_business_search)

----

```
* Search the shops nearby the user
  * GET /v1/search/nearby
  * Request
    * latitude: Latitude of a given location (decimal)
    * longitude: Longitude of a given location (decimal)
    * radius: Optional. Default is 5,000 meters (about 3 miles) (int)
  * Response
    * {
        "total": 10,
        "businesses": [{business object}]
      }

* APIs for businesses
  * GET /v1/businesses/:id
    * Return detailed information about a business
  * POST /v1/businesses
    * Add a business
  * PUT /v1/businesses/:id
    * Update details of a business
  * DELETE /v1/businesses/:id
    * Delete a business

```

## Data Model

### Read/Write Ratio

This is a read-heavy system and a relational database such as [MySQL](/mysql/README.md) is appropriate.

### Data Schema

> Business Table

| name | type |
|----|----|
| business_id | PK |
| address | |
| city | |
| state | |
| country | |
| latitude | |
| longitude | |

> Geo Index Table

| name | type |
|----|----|
| geohash | |
| business_id | |

| geohash | business_id |
| 32feac | 343 |
| 32feac | 343 |
| f31cad | 111 |
| f31cad | 112 |

## System Architecture Diagram

### Load Balancer

The load balancer routes incomming traffics across multiple services.

### LBS (Location Based Service)

It is a readonly stateless service.

### Business Service

Business owners add, update, or delete their own businesses. Usually write
operations and the QPS is not high.

Customers can view detailed information of the buinsesses. Usually read 
operations and the QPS is high.

### Database Cluster

AWS AuroraDB [MySQL](/mysql/README.md) is appropriate. That supports read replicas. 

### Scalability of Business Service and LBS

Business services and LBS services are stateless. If we use AWS ELB, Autoscaling
Group, we are able to scale out, scale in very easily.

## Algorithms To Find Nearby Businesses

### Two-Dimensional Search

This is very naive. This SQL is for search operation but it is not efficient
even though there are indexes of latitude, longitude. Because we need to use 2
dimensions for locations and there are lots of data for latitude, longitude
each.

```sql
SELECT business_id
       latitude,
       longitude
  FROM business
 WHERE (latitude BETWEEN {:my_latitude} - radius AND {:my_latitude} + radius) AND
       (longitude BETWEEN {:my_longitude} - radius AND {:my_longitude} + radius)
```

![](img/intersect_two_datasets.png)

If we use one dimension index we can improve the search speed.

![](img/different_types_of_geospatial_indexes.png)

### Evenly Divided Grid

This is about using fixed grids but there is hot partition problem.

### Geohash

We divide one square to 4 sub-squares recursively with following rules.

* Latitude range `[-90, 0]` is represented by `0`
* Latitude range `[0, 90]` is represented by `1`
* Longitude range `[-180, 0]` is represented by `0`
* Longitude range `[0, 180]` is represented by `1`

```
    0 1    1 1
    0 0    1 0
```

Geohash usually uses base-32 representation.

```
geohash of the Google headquater (length = 6)
1001 10110 01001 10000 11011 11010 (base-32 in binary)
9q9hvu (base-32)

geohash of the Facebook headquater (length = 6)
1001 10110 01001 10001 1000 10111 (base-32 in binary)
9q9jhr (base-32)
```

There are boundary issues. 

* `geohash: u000` and `geohash: ezzz` have no common prefix but they are close
  from each other.
  * This sql would fail to fetch all nearby businesses.
    ```sql
    SELECT * 
      FROM geohash_idx 
     WHERE geohash LIKE '9q8zn%'
    ```
* `geohash: 9q8znf` and `geohash: 9q8zn2` have long common prefix but they are
  far away from each other

When there not enough businesses we can increase the search radius with removing
digits from geohash.

### Quadtree

Quadtree is a data structure that partition a two-dimensional space by
recursively subdividing it into 4 parts until the leaf node has 100 busninesses.

This is a pseudo code of `buildQuadtree()`

```java
public void buildQuadtree(TreeNode node) {
    if (countNumberOfBusinessesInCurrentGrid(node) > 100) {
        node.subdivide();
        for (TreeNode child : node.getChildren()) {
            buildQuadtree(child);
        }
    }
}
```

These are data of leaf node

| Name | Size |
|--|--|
| Top left coordinates and bottom right coordinates | 32 bytes (8bytes * 4) |
| List of business IDs | 8 bytes * 100 (maximum number of businesses) |
| Total | 832 bytes |

There are data of internal node

| Name | Size |
|--|--|
| Top left coordinates and bottom right coordinates | 32 bytes (8bytes * 4) |
| Pointers to 4 children | 8 bytes * 4 |
| Total | 64 bytes |

These are estimation of memory

| Number | Description |
|--|--|
| 100 | The number of businesses in a leafnode |
| 2 million =~ 200 million / 100 | The number of leaf nodes |
| 0.67 million =~ 2 million * 1/3 | The number of internal nodes |
| 1.71 GB =~ 2 million * 832 bytes + 0.67 million * 64 bytes | The memory |

The size of the memory is `1.71 GB` and one server is ok.

It might take a few minutes to build the whole quadtree with 200 million businesses.
The time complexity is `O(N/100 log N/100)`

If we provide the coordinate of the user, we can return the specific leaf node
which has 100 businesses.

### Google S2

> * [가게 배달지역 관리방식 개편 프로젝트 | wooahan](https://techblog.woowahan.com/2717/)
> * [S2 library를 이용하여 가까운 위치의 사용자 찾기](https://yeongcheon.github.io/posts/2018-08-01-s2-geometry/)
> * [s2 geometry](https://s2geometry.io/)

----

Google S2 is complicated.

## Geohash vs Quadtree

Geohash

* It is simple to use and no need to build a tree
* It returns businesses within a specified radius.
* When the precision (level) of geohash is fixed, the size   of the grid is
  fixed.
* It is easy to update the index.

Quadtree

* It is complicated to implement because it needs to build the tree.
* Fetch k-nearest businesses.
* It can dynamically adjust the grid size based on density.
* It is complicated to udpate the index than geohash.

# High Level Deisgn Deep Dive

## Scale The Database

Business table is big and we can shard it by business_id.

Geospatial index table is small and we can save it in one server with read replicas.

## Caching

## Region and Availability Zones

## Filter Results by Time or Business Type

## Final Architecture Diagram
