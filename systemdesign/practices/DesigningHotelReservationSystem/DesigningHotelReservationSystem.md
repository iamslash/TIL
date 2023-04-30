- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
- [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [API Design](#api-design)
  - [High-Level Architecture](#high-level-architecture)
  - [Data Model](#data-model)
- [Low Level Design](#low-level-design)
- [System Extension](#system-extension)
- [Q\&A](#qa)
- [References](#references)

----

# Requirements

## Functional Requirement

* Show the hotel related page
* Show the hotel room-related detail page
* Reserver a room
* Admin console to add/remove/update hotel or room info
* Support the overbooking
* The price of a hotel room can be changed

## Non-Functional Requirement

* High concurrency
  * Many people can try to reserver a same room.
* Moderate latency
  * The latency should be small.

# Estimation

| Number | Description | Calculation |
|--|---|--|
| 5,000 | The number of hotels | |
| 1 Million | The number of rooms | |
| 70 % | occupied ratio | |
| 3 days | average stay duration | |
| 240,000 | daily reservation | 1 million x 0.7 / 3 = 233,333 |
| 3 | reservations per sec | 240,000 / 100,000 sec = 3 |
| 300 QPS | View hotel/room detail | |
| 30 QPS | Order booking page | |
| 3 QPS | Reserve rooms | |

# High Level Design

## API Design

```json
* Hotel APIs
  * GET /v1/hotels/<id>
  * POST /v1/hotels
  * PUT /v1/hotels/<id>
  * DELETE /v1/hotels/<id>

* Room APIs
  * GET /v1/hotels/<hotel-id>/rooms/<room-id>
  * POST /v1/hotels/<hotel-id>/rooms
  * PUT /v1/hotels/<hotel-id>/rooms/<room-id>
  * DELETE /v1/hotels/<hotel-id>/rooms/<room-id>

* Rservation APIs
  * GET /v1/reservations
  * GET /v1/reservations/<reserve-id>
  * POST /v1/reservations
    * Request
    {
        "startDate": "2022-03-01",
        "endDate": "2022-03-04",
        "hotelID": "333",
        "roomID": "U1122",
        "reservationID": "12231"
    }
  * DELETE /v1/reservations/<reserve-id>
```

## High-Level Architecture

## Data Model

# Low Level Design

# System Extension

# Q&A

# References
