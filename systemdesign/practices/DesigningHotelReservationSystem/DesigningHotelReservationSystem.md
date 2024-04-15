- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
  - [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [API Design](#api-design)
  - [Data Model Design](#data-model-design)
  - [High-Level Architecture](#high-level-architecture)
- [Design Deep Dive](#design-deep-dive)
  - [Reservation By Room Type](#reservation-by-room-type)
  - [Concurrency Issues](#concurrency-issues)
  - [Scalability](#scalability)
    - [Database Sharding](#database-sharding)
    - [Caching](#caching)
  - [Data Consistency Among Services](#data-consistency-among-services)
- [Interview Questions](#interview-questions)
- [References](#references)

----

# Requirements

## Functional Requirement

* Show the hotel related page.
* Show the hotel room-related detail page.
* Reserve a room.
* Admin console to add/remove/update hotel or room info.
* Support the overbooking.
* The price of a hotel room can be changed.

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
  * GET /v1/hotels/<hotel-id>
  * POST /v1/hotels
  * PUT /v1/hotels/<hotel-id>
  * DELETE /v1/hotels/<hotel-id>

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

## Data Model Design

* Hotel Service
  * hotel
    * hotel_id (PK)
    * name
    * address
    * location
  * room
    * room_id (PK)
    * room_type_id
    * floor
    * number
    * hotel_id
    * name
    * is_available

* Rate Service
  * room_type_rate
    * hotel_id (PK)
    * date (PK)
    * rate

* Reservation Service
  * reservation
    * reservation_id (PK)
    * hotel_id
    * room_id
    * srtart_date
    * end_date
    * status
    * guest_id

* Guest Service
  * guest
    * guest_id (PK)
    * first_name
    * last_name
    * email

There are **status** of **reservation** table 

* pending
* canceled
* paid
* rejected
* refunded

![](img/2024-04-14-21-46-55.png)

```
erDiagram
    HOTEL ||--o{ ROOM : "contains"
    HOTEL {
        string hotel_id PK
        string name
        string address
        string location
    }
    ROOM ||--|| HOTEL : "located in"
    ROOM {
        string room_id PK
        string room_type_id FK
        string floor
        string number
        string hotel_id FK
        string name
        boolean is_available
    }
    ROOM_TYPE_RATE ||--|| HOTEL : "applies to"
    ROOM_TYPE_RATE {
        string hotel_id PK
        date date PK
        decimal rate
    }
    RESERVATION }|--|| HOTEL : "booked in"
    RESERVATION }|--|| ROOM : "booked for"
    RESERVATION }|--|| GUEST : "booked by"
    RESERVATION {
        string reservation_id PK
        string hotel_id FK
        string room_id FK
        date start_date
        date end_date
        string status
        string guest_id FK
    }
    GUEST {
        string guest_id PK
        string first_name
        string last_name
        string email
    }
```

## High-Level Architecture

![](img/2023-05-12-21-48-40.png)

# Design Deep Dive

## Reservation By Room Type

Users can make a reservation not by `room_id` but by `room_type`. We can improve the design for reservating by room type. 

This is a improved API.

```json
* Rservation APIs
  * POST /v1/reservations
    * Request
    {
        "startDate": "2022-03-01",
        "endDate": "2022-03-04",
        "hotelID": "333",
        "roomTypeID": "23483727",
        "reservationID": "12231"
    }
```

These are improved database schemas.

* Hotel Service
  * hotel
    * hotel_id (PK)
    * name
    * address
    * location
  * room
    * room_id (PK)
    * room_type_id
    * floor
    * number
    * hotel_id
    * name
    * is_available

* Rate Service
  * room_type_rate
    * hotel_id (PK)
    * date (PK)
    * rate

* Reservation Service
  * room_type_inventory
    * hotel_id
    * room_type_id
    * date
    * total_inventory
    * total_reserved
  * reservation
    * reservation_id (PK)
    * hotel_id
    * room_id
    * srtart_date
    * end_date
    * status
    * guest_id

* Guest Service
  * guest
    * guest_id (PK)
    * first_name
    * last_name
    * email

![](img/2024-04-14-21-51-21.png)

```
erDiagram
    HOTEL ||--|| ROOM : "has"
    HOTEL {
        string hotel_id PK
        string name
        string address
        string location
    }
    ROOM {
        string room_id PK
        string room_type_id FK
        string floor
        string number
        string hotel_id FK
        string name
        boolean is_available
    }
    ROOM ||--o{ RESERVATION : "reserved by"
    ROOM_TYPE_RATE ||--|| HOTEL : "applies to"
    ROOM_TYPE_RATE {
        string hotel_id PK
        date date PK
        decimal rate
    }
    ROOM_TYPE_INVENTORY ||--|| HOTEL : "tracks inventory for"
    ROOM_TYPE_INVENTORY {
        string hotel_id FK
        string room_type_id FK
        date date
        int total_inventory
        int total_reserved
    }
    RESERVATION ||--|| GUEST : "made by"
    RESERVATION {
        string reservation_id PK
        string hotel_id FK
        string room_id FK
        date start_date
        date end_date
        string status
        string guest_id FK
    }
    GUEST {
        string guest_id PK
        string first_name
        string last_name
        string email
    }
```

These are access patterns.

Select rows within a date range.

```sql
SELECT date,
       total_inventory,
       total_reserved
  FROM room_type_inventory
 WHERE room_type_id = ${roomTypeId} AND
       hotel_id = ${hotelId} AND
       date BETWEEN ${startDate} and ${endDate}
```

Thie SQL returns data like this.

| date | total_inventory | total_reserved |
|--|--|--|
| 2021-07-01 | 100 | 97 |
| 2021-07-02 | 100 | 96 |
| 2021-07-03 | 100 | 95 |

For each entry, check the condition.

```c
if ((total_reserved + ${numberOfRoomsToReserve}) <= 110 % * total_inventory)
```

What if data is very big we can think two strategies.

* Use **hot sorage** for recent data, **cold storage** for old data.
* Database sharding. Shard key is `hotel_id`. The date will be sharded by `hash(hoteL_id) % number_of_db`.

## Concurrency Issues

Solutions for double booking problems of one users.

1. Prohibit double click on client-side.
2. Make APIs idempotent on same reservation.

Solutions for race condition of two users.

1. Pessimistic locking
2. Optimistic locking
3. Database constraints

These are SQLs for business logic.

```sql
-- step 1: Check room inventory
SELECT date,
       total_inventory,
       total_reserved
  FROM room_type_inventory
 WHERE room_type_id = ${roomTypeId} AND
       hotel_id = ${hotelId} AND
       date BETWEEN ${startDate} AND ${endDate}

if ((total_reserved + ${numberOfRoomsToReserved}) > 110% * total_inventory) {
  Rollback
}        

-- step 2: reserve rooms
UPDATE room_type_inventory
   SET total_reserved = total_reserved + ${numberOfRoomsToReserve}
 WHERE room_type_id = ${roomTypeId} AND
       date BETWEEN ${startDate} AND ${endDate} 

-- step 3: commit
Commit  
```

**Pessimistic locking**

Use `SELECT ... FOR UPDATE`. It will provide serializable isolation temporally. [isolation](/isolation/README.md#solution-of-non-repeatable-read-in-repeatable-read-isolation-level)

> Pros:

* It is easy to implement.
* It is suitable heavy contention data.

> Cons:

* Reduce system throughputs.
* Deadlocks may occur.

**Optimistic locking**

[Optimistic Locking](/systemdesign/README.md#optimistic-lock-vs-pessimistic-lock) is
faster than pessimistic locking.

> Pros:

* No need to lock the database.
* It is a good solution when data conflicts are rare.

> Cons:

* When data conflicts are so often it will reduce system throughputs.

The optimistic locking is a good option than Pessimistic locking when TPS is low.

**Database constraints**

```sql
CONSTRAINT `check_room_count` CHECK((`total_inventory - total_reserved` >= 0))
```

> Pros:

* Easy to implement.
* It is a good solution when data conflicts are rare.

> Cons:

* It is similar with [Optimistic Locking](/spring/SpringDataJpa.md#optimistic-locking).
* Contraint is not under control of SCM such as [git](/git/README.md).
* Not all database support constraints.
  
The constraint is a good option than Pessimistic locking when TPS is low.

## Scalability

The scalability is important when the QPS is `1,000` times higher than before like booking.com.

### Database Sharding

Shard data by `hotel_id % shard_num`. If QPS is `30,000` and the number of shards is `16`, Each shard handles `30,000 / 16 = 1,875` QPS.

- [Sharding](/systemdesign/README.md#sharding)

### Caching

Inventory cache is a good solution for room inventory write heavy system. [Redis](/redis/README.md) is a good solution.

```
key: hotelID_roomTypeID_{date}
val: the number of available rooms for the given hotel ID, room type ID and date
```

[Debezium](/Debezium/README.md) is a good solution for CDC from [MySQL](/mysql/README.md) to [Redis](/redis/README.md).

> Pros:

- Reduced database load.
- High Performance.

> Cons:

- The consistency between database and cache is difficult. We need to handle inconsistency. 

## Data Consistency Among Services

There are good solutions such as [2 Phase Commit](/distributedtransaction/README.md#2-phase-commit), [TCC](/distributedtransaction/README.md#tcc-try-confirmcancel) and [SAGAS](/distributedtransaction/README.md#saga).

# Interview Questions

- Design the hotel reservation system. The write QPS is 3.
- Design APIs.
- Design data models.
- How to handle concurrency issues for the same room reservation of one user?
  - Prohibiting double clicks.
- How to handle concurrency issues for the same room reservation of two users?
  - pessimistic lock
  - optimisitc lock
  - db constraint
- Explain about the isolation level of RDBMS.
- Explain MySQL locks.
  - Shared and Exclusive Locks
  - Intention Locks
  - Record Locks
  - Gap Locks
  - Next-Key Locks
  - Insert Intention Locks
  - AUTO-INC Locks
  - Predicate Locks for Spatial Indexes
- Design reservation by room types.
- What if QPS is 1,000 times higher like booking.com?
  - Sharding for write APIs
  - Caching for read APIs 
- How to handle distributed transactions?
  - 2 phase commit
  - SAGAS
- How does your system design handle high concurrency, specifically when multiple users attempt to book the same room?
  - The system utilizes optimistic and pessimistic locking mechanisms to manage concurrency. Optimistic locking is preferred when transaction conflicts are less frequent, whereas pessimistic locking can be used for high-conflict scenarios, ensuring that once a booking process begins, others cannot book until the transaction is complete.
- What strategies have you implemented in your API design to prevent overbooking in the system?
  - The system checks current reservations against total inventory and only allows a booking to proceed if it adheres to predefined limits, using database constraints or application logic to ensure no overbooking beyond a certain threshold (e.g., 110% of inventory).
- Can you describe the database sharding strategy used in this system? How does it help scale the application?
  - Database sharding is applied by dividing data among multiple databases using a hash of the hotel_id. This distributes the load and helps the system scale horizontally, allowing each shard to handle a portion of the traffic, thus enhancing performance.
- What caching mechanisms are employed to enhance system performance?
  - The system uses a caching layer, likely implemented with Redis, to store room availability information, which decreases the load on the primary database and provides faster access to data, improving response times for end-users.
- How do you ensure data consistency across services in a distributed architecture?
  - The system could employ techniques like the Two-Phase Commit protocol, TCC (Try-Confirm/Cancel), or SAGAs for managing distributed transactions, ensuring consistency across different services by coordinating transactions that span multiple services.
- Explain the role of room type inventory in your database schema. How does it facilitate the booking process?
  - The room type inventory table tracks available and reserved rooms by date and room type, allowing the system to query and update inventory efficiently. This is crucial for allowing bookings by room type rather than individual room, simplifying the reservation process for users.
- Discuss the API design changes proposed for allowing reservations by room type instead of specific room IDs.
  - The modified API design lets users book rooms based on type, providing more flexibility. This involves checking aggregated inventory for a room type rather than individual rooms, which can streamline the user experience and improve backend efficiencies.
- How would you handle a sudden spike in traffic, say 1,000 times the normal load, as seen on platforms like Booking.com?
  - To manage such a spike, sharding and caching would be critical. Sharding ensures that the database load is distributed, while caching reduces the number of queries hitting the database directly, handling high read demands effectively.
- What are the potential downsides of using database constraints to manage room reservations?
  - While database constraints ensure that overbooking doesn't occur, they are not flexible, hard to manage as part of application deployment, and not supported by all database systems. They can also lead to frequent transaction rollbacks if conflicts are common.
- If two users simultaneously try to book the last available room of a particular type, how does your system resolve this conflict?
  - The system could use either optimistic or pessimistic locking to manage this. With optimistic locking, both users may attempt to book, but the final commit checks will ensure only one succeeds. Pessimistic locking would lock the booking for the first user until their transaction completes, preventing the second user from booking until the lock is released.

# References

* [RestAppHotelbooking | java](https://github.com/BogushAleksandr/RestAppHotelbooking)
* [Booking Application | go](https://github.com/revel/examples/blob/master/README.md)
* [Hotel Reservation Management System | python](https://github.com/rub9542/Hotel-Reservation-Management-System)
