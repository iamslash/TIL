- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-functional Requirements](#non-functional-requirements)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
  - [Traffic Estimates](#traffic-estimates)
  - [Storage Estimates](#storage-estimates)
  - [Network](#network)
- [System APIs](#system-apis)
- [High-level Architecture](#high-level-architecture)
- [Low-level Architecture](#low-level-architecture)
- [System Extentions](#system-extentions)
- [Q&A](#qa)
- [References](#references)

-----

# Requirements

## Functional Requirements

* Drivers notify the system their location and availability for passengers.
* Passengers check available drivers near around them.
* Customers request a ride and neary by drivers are notified.
* Once a driver and customer accept a ride, they can see each other's location, until the trip is finished.
* When they arrive the destination, the driver marks the ride completed and become available for the next ride.

## Non-functional Requirements

* Latency should be under 100 ms.
* The system should be high availble.

# Capacity Estimation and Constraints

## Traffic Estimates

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 300 M customers | RU of customers |
| 1 M drivers | RU of drivers |
| 1 M customers | DAU of customers |
| 500 K drivers | DAU of drivers |
| 3 per sec | drivers notify location per sec |

## Storage Estimates

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 3 bytes | driver_id |
| 8 bytes | latitude, longitude |
| 35 byets (3 + 8 + 8 + 8 + 8) | value of DriverLocationHashTable. `{driver_id:{old_latitude,old_longitude,new_latitude,new_longitude}}` |
| 35 MB (1 M * 35 bytes) | Hash Table for driver locations  |
| 5 customers | average subscribed customers for one driver |
| 8 bytes | customer_id |
| 21 MB (500 K * 3 bytes (driver_id) + 500K * 5 average subscribers * 8 bytes) | Hash Table of subscribers for drivers |

## Network


| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 2.5 M (5 * 500 K) | total subscribers of drivers |
| 47.5 MB/s (2.5 M * 19 bytes) | egress of drivers' locations per sec |

# System APIs

```
view_drivers(api_key)

request_ride(api_key)

accept_driver(api_key)

cancel_driver(api_key)
```

```
accept_customer(api_key)

notify_driver_location(api_key)
```

# High-level Architecture

# Low-level Architecture

* The system should update Quad Tree every 10 seconds.
* The system store DriverLocationHashTable with `{driver_id:{old_latitude,old_longitude,new_latitude,new_longitude}}`.
  * driver_id : 3 bytes (RU of driver is 1 M)
  * others are 8 bytes
  * The size of value is 35 bytes without hash table overhead.
  * Driver Location Server will hold DriverLocationHashTable.
* Driver Notification Server will notify customers' location to drivers.
* Customer Notification Server will notify drivers' location to customers.
* 

# System Extentions

# Q&A

# References

* [UBER system design @ medium](https://medium.com/@narengowda/uber-system-design-8b2bc95e2cfe)
* [System Design Interview: mini Uber @ medium](https://medium.com/@eileen.code4fun/system-design-interview-mini-uber-a48444258402)
  