- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
  - [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [API Design](#api-design)
  - [Data Model](#data-model)
  - [High-Level Architecture](#high-level-architecture)
- [High Level Design Deep Dive](#high-level-design-deep-dive)
- [References](#references)

----

# Requirements

## Functional Requirement

* Show the hotel related page
* Show the hotel room-related detail page
* Reserve a room
* Admin console to add/remove/update hotel or room info
* Support the overbooking
* The price of a hotel room can be changed

## Non-Functional Requirement

* High concurrency
  * Many people can try to reserve a same room.
* Moderate latency
  * The latency should be small.

## Estimation

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

## Data Model



## High-Level Architecture

# High Level Design Deep Dive

# References

* [RestAppHotelbooking | java](https://github.com/BogushAleksandr/RestAppHotelbooking)
* [Booking Application | go](https://github.com/revel/examples/blob/master/README.md)
* [Hotel Reservation Management System | python](https://github.com/rub9542/Hotel-Reservation-Management-System)
