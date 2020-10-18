- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-functional Requirements](#non-functional-requirements)
- [System APIs](#system-apis)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
  - [Traffic Estimation](#traffic-estimation)
  - [Storage Estimation](#storage-estimation)
  - [Bandwith Estimation](#bandwith-estimation)
  - [High-level Estimation](#high-level-estimation)
- [High-level Architecture](#high-level-architecture)
  - [Workflow](#workflow)
- [Low-level Architecture](#low-level-architecture)
  - [Messages Handling](#messages-handling)
  - [How will clients main an open connection with the server?](#how-will-clients-main-an-open-connection-with-the-server)
  - [How can the server keep track of all the opened connection to redirect messages to the users](#how-can-the-server-keep-track-of-all-the-opened-connection-to-redirect-messages-to-the-users)
  - [What will happend when the server receives a message for a user who has gone offline ?](#what-will-happend-when-the-server-receives-a-message-for-a-user-who-has-gone-offline-)
  - [How many chat server we need?](#how-many-chat-server-we-need)
  - [How to know which server hold the connection to which user?](#how-to-know-which-server-hold-the-connection-to-which-user)
  - [How should the server process a 'deliver message' request?](#how-should-the-server-process-a-deliver-message-request)
  - [How does the messenger maintain the sequencing of the messages?](#how-does-the-messenger-maintain-the-sequencing-of-the-messages)
  - [Which storage system we should use ?](#which-storage-system-we-should-use-)
  - [How should clients eiffiently fetch data form the server ?](#how-should-clients-eiffiently-fetch-data-form-the-server-)
  - [Managing user's status](#managing-users-status)
  - [Partioning Database](#partioning-database)
- [System Extentions](#system-extentions)
  - [Cache](#cache)
  - [Load balancing](#load-balancing)
  - [Fault tolerance and replication](#fault-tolerance-and-replication)
- [Q&A](#qa)
- [References](#references)

-----

# Requirements

## Functional Requirements

* The system supports 1on1 between users.
* The system supports group chats.
* The system supports push notifications.
* The system supports to keep track of online/offline statuses of users.
* The system supports persistent storage of char history.

## Non-functional Requirements

* The system supports real-time chat experience with minimum latency.
* The system supports highly-consistent, users can the same chat history on all their devices.
* The system supports highly-availabilty, but lower availability in the interest of consistency.

# System APIs

```
sendMessage(api_key, msg)

recvMessage(msg, from_user)
```

# Capacity Estimation and Constraints

## Traffic Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 500 M   | DAU (Daily Active Users) |
| 40 | messages per day for each user |
| 20 billion | messager per day  for all users |

## Storage Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 100 bytes | average data size for one message   |
| 2 TB / day (20 billion * 100 bytes) | storage per day  |
| 3.6 PB (2 TB/day * 365 days * 5 years) | storage for 5 years |

## Bandwith Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 25 MB / day (2 TB / 86400 sec) | ingress data size per sec |
| 25 MB / day (2 TB / 86400 sec) | egress data size per sec |

## High-level Estimation

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 2 TB | storage for each day  |
| 3.6 PB | storage for 5 years  |
| 25 MB | ingress data  |
| 25 MB | egress data |

# High-level Architecture

![](DesigningFacebookMessengerHighLevelArch.png)

## Workflow

* User-A sends a message to User-B through the chat-A server.
* The chat-A server received the message and sends an ack to User-A.
* The chat-A server stores the message in its database and sends the message to User-B through the chat-B server.
* User-B receives the message and sends the ack to the chat-B server.
* The chat-A server notifies to User-A that the message has been delivered successfully to User-B

# Low-level Architecture

## Messages Handling

Pull vs Push

## How will clients main an open connection with the server?

Long Polling, WebSockets

## How can the server keep track of all the opened connection to redirect messages to the users

hash map

## What will happend when the server receives a message for a user who has gone offline ?

Store messages and send later

## How many chat server we need?

50K connections for one server. Finally 10K servers for 500 million connectsions a day

## How to know which server hold the connection to which user?

hash map for {user_id : server_ip}

## How should the server process a 'deliver message' request? 

* Store the message in the DB
* Send the messag to the receiver
* Send an ack to the sender
* Storing, Sending can be done in background.

## How does the messenger maintain the sequencing of the messages?

Use time stamp in message.

* User-1 sends a message M1 to the Chat-A server for User-2.
* The Chat-A server receives M1 at T1.
* User-2 ends a message M2 to the Chat-B server for User-1.
* The Chat-B server receives M2 at T2. such that T2 > T1.
* Chat servers sends M1 to User-2 and M2 to User-1.

User-1 will see M1 first and then M2, whereas User-2 will see M2 first and then M1.
Client adjust the order of messages by time stamp or message sequnces.

## Which storage system we should use ?

MySQL, MongoDB is not a good solution. HBase is a good solution.

## How should clients eiffiently fetch data form the server ?

Pagination is a good solution.

## Managing user's status

* When User-1 logins, it pull status of it's friends.
* When User-1 sends to User-2 that has offline, Chat-A sends a ack with failure and update the status of User-2.
* When User-2 logins, Chat-B broadcast User-2's status to it's friends.
* User-1 can pull the status from Chat-A periodically.

## Partioning Database

* Partitioning based on user_id is a good solution.
* Partitioning based on message_id is not a good solution because of slow fetching time.

# System Extentions

## Cache

Cache recent messages (15 mins).

## Load balancing

* in front of Dispatch server.

## Fault tolerance and replication

* What will happend when a chat server fails ?
  * Clients will reconnect.
* Should we store multiple copies of user messages ?
  * Reed-Solomon encoding ???

# Q&A

# References
