- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-functional Requirements](#non-functional-requirements)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
  - [Traffic Estimates](#traffic-estimates)
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

# System APIs

# High-level Architecture

# Low-level Architecture

# System Extentions

# Q&A

# References

* [UBER system design @ medium](https://medium.com/@narengowda/uber-system-design-8b2bc95e2cfe)
* [System Design Interview: mini Uber @ medium](https://medium.com/@eileen.code4fun/system-design-interview-mini-uber-a48444258402)
  