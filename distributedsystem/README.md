- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [What is distributed systems](#what-is-distributed-systems)
- [Paritioning, Replication](#paritioning-replication)
- [Timing/Ordering Problems](#timingordering-problems)
- [Consensus Problems](#consensus-problems)
- [The FLP Impossibility Result](#the-flp-impossibility-result)
- [CAP Theorem](#cap-theorem)
- [Consistency Models](#consistency-models)
- [Vector Clock](#vector-clock)
- [2 Phase Commit](#2-phase-commit)
- [Paxos](#paxos)
- [Raft](#raft)
- [CRDT's (convergent replicated data types)](#crdts-convergent-replicated-data-types)
- [CALM (consistency as logical monotonicity)](#calm-consistency-as-logical-monotonicity)

-----

# Abstract

Describe distributed systems.

# Essentials

[6.824: Distributed Systems](6.824.md)

# Materials

* [Distributed systems for fun and profit](https://book.mixu.net/distsys/)
* [6.824: Distributed Systems](https://pdos.csail.mit.edu/6.824/)
  * [schedule](https://pdos.csail.mit.edu/6.824/schedule.html)
  * [video](https://www.youtube.com/channel/UC_7WrbZTCODu1o_kfUMq88g/playlists)
* [Distributed Systems in One Lesson by Tim Berglund](https://www.youtube.com/watch?v=Y6Ev8GIlbxc)
* [Principles Of Microservices by Sam Newman](https://www.youtube.com/watch?v=PFQnNFe27kU)

# What is distributed systems

A distributed system is a collection of independent computers or nodes
interconnected through a network, working together to perform tasks or achieve
common goals as a single, cohesive system. In distributed systems, the computers
or nodes communicate and coordinate their actions by passing messages to each
other. Each node in the distributed system has its own local memory, and the
system operates by sharing information among the various nodes to solve problems
collectively.

Distributed systems are designed to achieve several key objectives, such as:

* **Scalability**: Distributed systems allow for horizontal scaling, where
additional resources can be added by increasing the number of nodes in the
system. This enables the systems to handle higher workloads, large amounts of
data, and growing user bases without compromising performance.

* **Fault tolerance**: By distributing the workload and data across multiple
nodes, distributed systems can remain operational even in the face of partial
failures, such as hardware crashes, network outages, or software bugs. The
system's ability to recover from failures and continue functioning is a critical
feature of distributed systems.

* **High availability**: Distributed systems are designed to keep services
continuously operational and accessible to users, often by using replication and
redundancy techniques. This helps ensure that the system remains available even
during unexpected events or maintenance periods.

* **Resource sharing**: Distributed systems enable sharing of resources such as
computing power, storage, and network bandwidth among the nodes to accomplish
tasks collectively and efficiently.

* **Geographic distribution**: In some cases, distributed systems can span
across large geographical areas, providing low-latency access to resources,
data, or services for users in diverse locations.

Examples of distributed systems include web services and applications,
peer-to-peer networks, distributed databases, blockchain networks, distributed
file systems, and cloud computing platforms.

Designing and maintaining distributed systems is complex due to several
challenges such as handling **communication**, **concurrency**,
**synchronization**, **consistency**, **fault tolerance**, and **security**.
Various algorithms and techniques, such as **consensus algorithms**,
**partitioning**, **replication**, and **caching**, are employed to address
these challenges and ensure the correct functioning of distributed systems.

# Paritioning, Replication

Partitioning and replication are two key techniques employed in distributed
systems to achieve **scalability**, **fault tolerance**, and **high
availability**. Both of
these techniques are used to manage the distribution of data and workload among
multiple nodes or servers in distributed systems. Let us discuss these
techniques in detail.

**Partitioning**:

**Partitioning**, also known as **sharding** or **data partitioning**, is the
process of dividing a large dataset into smaller, manageable pieces (partitions)
and distributing them across multiple nodes or servers in a distributed system.
The primary goal of partitioning is to enable **horizontal scaling** and reduce
the load on a single node, allowing the system to handle larger amounts of data
and increased workload.

There are various partitioning strategies that can be applied in distributed
systems, such as:

* **Range Partitioning**: Data is partitioned based on a range of values. For
  example, in a distributed database, we can partition records based on the
  timestamp or the value of a particular field.
* **Hash Partitioning**: A hash function is applied to a specific attribute of
  the data, and the resulting hash value determines the partition it belongs to.
  This aids in distributing the data evenly across the nodes and prevents
  hotspots.
* **List Partitioning**: Data is partitioned based on a list of predefined
  values. Each partition node is assigned a set of unique values, and any data
  matching these values is stored in that partition.
* **Round-robin Partitioning**: Data is distributed evenly across all nodes in a
  cyclic manner. This can help in achieving a balanced distribution of data
  across all nodes.

**Replication**:

**Replication** is the process of creating multiple copies of the same data
across different nodes or servers in a distributed system. Replication mainly
focuses on fault tolerance and high availability, ensuring that the system
remains reliable and accessible even in the face of node failures.

There are various replication techniques and strategies used in distributed
systems, such as:

* **Synchronous Replication**: In this technique, any write operation is
  considered successful only after the data is written to all replicas across
  the distributed system. While this provides strong consistency and better fault
  tolerance, it could result in increased latency due to the need for
  acknowledgments from all replicas.
* **Asynchronous Replication**: Here, the write operation is considered
  successful as soon as the data is written to the primary node. The data is
  then asynchronously propagated to other replicas. Asynchronous replication
  offers lower latency and better performance, but may result in temporary
  inconsistencies among replicas in case of node failures.
* **Active-Passive Replication**: In this strategy, only the primary node
  (active) serves read and write requests, while other replica nodes (passive)
  remain on standby and synchronize their data with the primary node. In case
  the primary node fails, one of the passive nodes can be promoted to serve as
  the new primary node.
* **Active-Active Replication**: All nodes participate in serving read and write
  requests, enabling load balancing, and better system performance. However,
  this may result in increased complexity when maintaining consistency among
  replicas.

In conclusion, **partitioning** and **replication** are crucial components in
the design of **distributed systems**. **Partitioning** enables systems to
manage **a large amount of data** and **workload** among **multiple nodes**
effectively, while replication enhances **fault tolerance** and **availability**
of the system. The combination of these techniques allows distributed systems to
be more scalable, reliable, and efficient in coping with increasing demands.

# Timing/Ordering Problems

In distributed systems, the timing and ordering of events, messages, or
operations across multiple nodes are crucial to ensure their correct functioning
and consistency. However, achieving accurate timing and ordering can be quite
challenging due to the inherent complexities and variability of distributed
systems. When independent nodes communicate only by exchanging messages across a
network, there is no global clock or shared memory to provide a precise
universal time reference or event ordering.

Timing and ordering problems in distributed systems arise primarily due to:

* **Network latency**: The time it takes for a message to transmit between nodes
can vary due to network conditions, congestion, or physical distance. This
variability in message transmission times can lead to nodes having different
perspectives of event order.

* **Clock skew**: The clocks of individual nodes in a distributed system may
  drift and become unsynchronized over time due to clock imperfections, leading
  to discrepancies in timestamps and difficulty in accurately determining the
  order of events.

* **Concurrency**: In distributed systems, multiple nodes can perform operations
  concurrently, making it challenging to determine the relative ordering of
  events and ensure consistent system behavior.

These issues can lead to complications, such as:

* **Inconsistency in replicated data**: In distributed systems that store
  multiple copies of data across different nodes, the timing and ordering of
  updates are crucial to maintaining data consistency. If update messages arrive
  in different orders at different nodes, the replicas can become inconsistent.

* **Violation of causality in message ordering**: The causal order of events
  defines the cause-and-effect relationships between them. Preserving causal
  order is essential to prevent anomalies, such as sending a reply to a message
  before receiving the message itself.

* **Difficulties in global state and snapshot collection**: Accurate timing and
  ordering information is essential to obtain a consistent global snapshot of a
  distributed system at a given point in time. Without carefully considering
  timing and ordering, partial snapshots might be taken at inconsistent times,
  resulting in a distorted view of the overall system.

Several approaches have been proposed to solve timing and ordering issues in distributed systems:

* **Logical clocks**: **Lamport timestamps** and **vector clocks** provide mechanisms
  for tracking causal order and understanding the relative ordering of events
  without relying on synchronized physical clocks.

* **Synchronization protocols**: Techniques like **Network Time Protocol (NTP)**
  and **Precision Time Protocol (PTP)** help synchronize node clocks in
  distributed systems to provide a common time reference.

* **Consensus algorithms**: **Paxos**, **Raft**, and other consensus mechanisms help
  nodes agree on the order of events, ensuring a consistent view and proper
  behavior in distributed systems despite the lack of a global clock.

* **Message ordering techniques**: Techniques like total order broadcast or FIFO
  (first-in, first-out) channels ensure that messages are ordered and processed
  consistently across all nodes in the system.

Addressing these timing and ordering challenges is critical to guarantee the
consistency, reliability, and correct functioning of distributed systems.

# Consensus Problems

The consensus problem is a fundamental challenge in distributed computing, where
multiple processes or nodes need to reach an agreement on a common value based
on their individual inputs. This problem often arises in practical scenarios,
such as distributed transactions, distributed databases, leader election, or
state machine replication, where nodes in a distributed system need to agree on
values or actions to maintain overall consistency and correct functioning of the
system.

A consensus algorithm aims to solve the consensus problem while satisfying the
following crucial properties:

* **Agreement**: All non-faulty nodes must agree on the same value.
* **Validity**: The agreed-upon value must be the input value of one of the
  nodes.
* **Termination**: All non-faulty nodes must eventually decide on a value.
* **Integrity**: A node must decide on a value at most once, and if it decides
  on a value, it must be the value it initially proposed or received from
  another node.
  
The consensus problem becomes even more challenging in the presence of faulty
nodes that may fail, crash, or exhibit Byzantine behavior (where nodes may
exhibit unpredictable, contradictory, or malicious behavior). **Consensus
algorithms** need to tolerate a certain number of faulty nodes while still
guaranteeing the desired properties.

Some well-known consensus algorithms and protocols include:

* **Paxos**: A consensus algorithm that tolerates non-Byzantine faults (such as
  node crashes) and guarantees safety (agreement and validity) and liveness
  (termination) in partially synchronous systems.
* **Raft**: A consensus algorithm specifically designed for readability and
  understandability, offering similar guarantees as Paxos, but with a more
  straightforward approach and implementation.
* **Byzantine fault-tolerant (BFT) protocols**: Algorithms like Practical
  Byzantine Fault Tolerance (PBFT) and Stellar Consensus Protocol (SCP) provide
  consensus guarantees in the presence of Byzantine faults, where nodes may
  exhibit arbitrary or malicious behavior.
* **Proof-of-Work (PoW) and Proof-of-Stake (PoS)**: Consensus mechanisms used in
  cryptocurrency systems like Bitcoin and Ethereum to achieve agreement on the
  global ledger state in a decentralized manner.

Solving the consensus problem is essential for ensuring the **accuracy**,
**consistency**, and **reliability** of distributed systems. It allows them to
coordinate their actions effectively despite inherent challenges such as network
latency, node failures, and asynchrony.

# The FLP Impossibility Result

**The FLP Impossibility Result**, named after its authors Michael J.
**F**ischer, Nancy A. **L**ynch, and Michael S. **P**aterson, is a fundamental
theorem in distributed computing that demonstrates the inherent limitation of
achieving consensus in a completely asynchronous distributed system. The FLP
Impossibility Result, proven in their 1985 paper titled "Impossibility of
Distributed Consensus with One Faulty Process," states that it's impossible to
design a consensus algorithm that guarantees **agreement**, **validity**, and
**termination** in the presence of even a single faulty process within an
asynchronous system.

# CAP Theorem

The CAP theorem, also known as **Brewer's theorem**, is an important principle
in distributed systems, proposed by Eric Brewer in 2000 and later proven by Seth
Gilbert and Nancy Lynch in 2002. The CAP theorem states that it is impossible
for a distributed data store to simultaneously provide all three of the
following guarantees:

* **Consistency (C)**: Every read operation on the data store returns the most
  recent write or an error. This means that all nodes see the same data at the
  same time, ensuring a consistent view of the data throughout the system.
* **Availability (A)**: Every request (read or write) made to the data store is
  completed successfully without any errors and within a reasonable time,
  regardless of the state of individual nodes in the system.
* **Partition Tolerance (P)**: The system continues to function correctly even
  when there are network partitions or communication breakdowns between nodes in
  the distributed system.

The CAP theorem expresses the inherent trade-offs and limitations when designing
distributed systems. According to the theorem, a distributed data store can
satisfy at most two out of the three guarantees. Thus, designers of distributed
systems must choose which two properties to prioritize, based on the specific
requirements and constraints of their use cases.

This leads to three possible combinations of these guarantees, also known as the
CAP configurations:

* **CP (Consistency and Partition Tolerance)**: In this configuration, the
  system maintains data consistency and can tolerate network partitions, but it
  might sacrifice availability during a network split or when a node is
  unreachable.
* **AP (Availability and Partition Tolerance)**: The system prioritizes
  availability and partition tolerance, meaning it will always return a response
  even if there are network partitions or node failures. However, it might
  sacrifice strong consistency, and nodes may temporarily have a conflicting or
  outdated view of the data.
* **CA (Consistency and Availability)**: The system maintains both consistency
  and availability but cannot tolerate network partitions. Since distributed
  systems with multiple nodes are inherently prone to partitions, this
  configuration is often considered unrealistic in the context of the CAP
  theorem.

In practical distributed system design, the specific requirements and
performance expectations of the use case often determine which properties are
prioritized. For example, some systems may require strict consistency and be
able to sacrifice availability during network partitions (e.g., distributed
databases), while others may prioritize data availability with weaker
consistency guarantees (e.g., caching systems, eventual consistency models). The
CAP theorem helps distributed system designers make informed decisions about
these trade-offs to build scalable and reliable applications.

# Consistency Models

Consistency models describe the level of guarantees provided by a distributed
system to ensure that multiple nodes have a standard view of the data. These
models are essential for maintaining data coherence and correctness when data is
accessed, updated, or replicated across different nodes. The three primary
consistency models include **strong consistency**, **weak consistency**, and **eventual
consistency**.

**Strong Consistency**: Strong consistency guarantees that all nodes in a
distributed system have a consistent view of the data at all times. As soon as
an update or write operation is performed on the data, every subsequent read
operation on any node will return the updated value. In other words, all nodes
always see the most up-to-date version of the data.

Strong consistency ensures that the system behaves as if there is only one copy
of the data, allowing for a simple, intuitive interaction with the system.
However, strong consistency often comes at the cost of increased latency,
reduced availability, and decreased scalability, as the system may need to wait
for synchronization and communication among nodes before proceeding with read or
write operations.

Examples of strong consistency models include linearizability, sequential
consistency, and strict consistency.

**Weak Consistency**: Weak consistency models provide no strict guarantees on
when the updates are propagated to all nodes or when the data becomes consistent
across the system. In weakly consistent systems, read operations can return
outdated or conflicting values until the update propagates to all nodes. This
allows for reduced latency and higher availability, but at the cost of providing
temporary inconsistent views of the data.

Since weak consistency models do not impose stringent constraints on data
propagation, they can be more performant and scalable than strong consistency
models. However, they require careful design and management to ensure that
applications can handle the uncertainty and inconsistencies that may arise.

**Eventual Consistency**: Eventual consistency is a specific form of weak
consistency that guarantees that if no new updates occur, all nodes in the
distributed system will eventually converge to the same data state. In other
words, given a period without any updates, all nodes will eventually have a
consistent view of the data.

Eventual consistency allows for faster write and read operations, better
availability, and higher fault tolerance, as it does not require immediate
propagation or acknowledgement of updates across nodes. However, during the
period when updates are being propagated, the system might exhibit temporary
inconsistencies.

Eventual consistency is often suitable for applications where temporary
inconsistencies or out-of-date data can be tolerated, such as social media
feeds, caching systems, or distributed DNS systems.

In summary, strong consistency models provide the highest level of guarantees
for data consistency but often come with performance trade-offs, including
increased latency and reduced availability. Weak consistency and eventual
consistency models trade some data consistency guarantees for better
performance, scalability, and availability. System designers must carefully
consider these trade-offs and choose the appropriate consistency model based on
their application requirements and use cases.

# Vector Clock

A vector clock is a mechanism in distributed systems for tracking and ordering
events based on causal relationships, without using synchronized physical
clocks. Each node maintains a vector of integer counters, representing other
nodes. Nodes update vectors when events occur and exchange vectors during
message passing. This helps maintain consistency and detect conflicts by
comparing vector clocks to determine event ordering.

[vector clock | wikipedia](https://en.wikipedia.org/wiki/Vector_clock)

# 2 Phase Commit

**Two-Phase Commit (2PC)** is a distributed transaction protocol used to ensure
the atomicity of a transaction across multiple nodes in a distributed system.
This protocol coordinates all nodes participating in a transaction to decide
whether to **commit (apply the changes)** or **abort (discard the changes)** the
transaction globally. The main goal of the 2PC protocol is to ensure that all
nodes either commit the transaction or roll it back, with no partial or
inconsistent outcomes.

The Two-Phase Commit protocol consists of two primary phases: the **voting phase**
and the **commit phase**.

**Voting Phase** (also called the Prepare Phase): 

* The node initiating the transaction, usually called the **coordinator**, sends
  a "prepare" message to all participating nodes, known as **cohorts**. These
  cohorts are responsible for executing parts of the transaction.
* Each **cohort** receives the "prepare" message and checks whether it can
  proceed with the transaction. If the **cohort** can commit its part of the
  transaction, it responds with a "yes" vote and writes an entry to its logs to
  prepare for the commit. If the **cohort** cannot commit, it responds with a
  "no" vote and aborts its part of the transaction.

**Commit Phase**: 

* The **coordinator** collects votes from all **cohorts** and makes a decision
  based on the voting results.
  * If all cohorts vote "yes", the coordinator decides to commit the transaction
    globally, writes an entry to its logs, and sends a "commit" message to all
    cohorts.
  * If any cohort votes "no" or there's a communication failure, the coordinator
    decides to abort the transaction, writes an abort entry to its logs, and
    sends an "abort" message to all **cohorts**.
* **Cohorts** receive the coordinator's decision and act accordingly. If they
  receive a "commit" message, they commit their parts of the transaction and log
  the commit. If they receive an "abort" message, they rollback or discard their
  changes and log the abort.

The Two-Phase Commit protocol ensures that a distributed transaction is atomic
(all-or-nothing) across all participating nodes. However, it has some drawbacks.
It is a blocking protocol that requires all participating nodes to wait for the
coordinator's decision, which can lead to performance issues and reduced
availability. Also, it assumes that the **coordinator** and **cohorts** do not
fail during the commit process; handling such failures may add complexity to the
protocol. Alternatives like **Three-Phase Commit (3PC)** and consensus protocols
like **Paxos** and **Raft** address some of these limitations.

# Paxos

> [Paxos](https://www.microsoft.com/en-us/research/uploads/prod/2016/12/paxos-simple-Copy.pdf)

The **Paxos** protocol is a consensus algorithm used in distributed systems to
ensure that a set of nodes can agree on a single value, even in the presence of
node failures, crashes, or network latency. Paxos is designed to be
fault-tolerant, ensuring that the consensus is reached as long as a majority of
nodes are functioning correctly and communicating. It is commonly used in
applications like distributed databases, state machine replication, and
distributed transaction management, where multiple nodes need to agree on
consistent values or operations.

The key idea behind **Paxos** is that a value is considered agreed upon if a
majority of nodes have accepted it. By selecting values in increasing order of
proposal numbers, the protocol ensures that values are chosen by a unique
proposer, and that this chosen value eventually becomes the agreed-upon value.

The Paxos protocol involves three primary roles:

* **Proposers**: Nodes that initiate proposals for a value to be agreed upon.
* **Acceptors**: Nodes voting on proposals, accepting or rejecting them based on
  Paxos rules.
* **Learners**: Nodes that learn about the chosen value and take appropriate
  actions. 
  
Paxos operates in two main phases:

* **Prepare Phase**:
  * A proposer selects a proposal number n and sends a "prepare request" message
    to a majority of acceptors.
  * When an acceptor receives the prepare request, it checks if the proposal
    number n is higher than any proposal it has seen before. If so, the acceptor
    acknowledges the message, promising not to accept proposals with a lower
    number than n, and sends back its highest accepted proposal (if any) to the
    proposer.
* **Accept Phase**:
  * The proposer receives responses from a majority of acceptors. It selects the
    value associated with the highest-numbered proposal from the responses (or
    uses a new value if no previously accepted proposals exist) and sends an
    "accept request" with the pair (n, value) to a majority of acceptors.
  * If an acceptor receives the accept request and has not promised to ignore
    proposals with numbers less than n, it accepts the proposal and sends an
    "accepted" message to the proposer and learners.

When **proposers** and **learners** receive "accepted" messages from a majority
of **acceptors**, the value is considered chosen, and consensus is achieved.

**Paxos** guarantees **safety** (ensuring that no two nodes can agree on
different values) under all conditions and guarantees **liveness** (ensuring
that a value is eventually agreed upon) in the absence of failures or
asynchrony. However, the protocol can be complex to understand and implement.
Variants like **Multi-Paxos** and other consensus algorithms like **Raft** have
been developed to address some complexities and improve upon certain aspects of
the basic Paxos protocol.

# Raft

> * [Raft 분산 합의 알고리즘과 Python에서의 활용 @ NAVERD2](https://d2.naver.com/helloworld/5663184)
>  * [The Raft Consensus Algorithm](https://raft.github.io/)
>  * [raft scope animation](https://raft.github.io/raftscope/index.html)
>  * [pyraft](https://github.com/lynix94/pyraft)
> * [Raft Consensus Algorithm](https://swalloow.github.io/raft-consensus/)

---

Raft is a consensus algorithm designed for distributed systems to ensure that a
set of nodes can agree on a single value or sequence of values, even in the
presence of node failures, crashes, or network latency. **Raft** was designed as
an alternative to the **Paxos** protocol, with a focus on simplicity and
understandability, while providing similar **safety** and **liveness**
guarantees.

The **Raft** algorithm is primarily used for managing replicated state machines,
distributed databases, and distributed transaction management, where multiple
nodes need to agree on consistent values or operations.

The main components of the Raft algorithm include:

* **Leader Election**: Raft divides time into terms, and for each term, there is
  a single elected leader responsible for managing the system and processing
  client requests. Nodes (also called candidates) can initiate a leader election
  if they do not receive messages from the leader within a certain time window.
  The election process involves sending vote requests to other nodes, and
  eventually, a candidate becomes the leader when it receives votes from the
  majority of nodes.

* **Log Replication**: Once a leader is elected, it is responsible for managing
  and replicating the log of operations (or commands) to other nodes (also
  called followers). The leader appends new entries to its log and sends them to
  followers. Followers acknowledge when they successfully append entries to
  their logs.

* **Log Commit**: The leader determines when an entry is considered "committed",
  which means it has been replicated safely across a majority of nodes. A
  committed entry is applied to the state machine, and its result is returned to
  the client. The leader communicates the commit index to followers so they can
  apply the committed entries locally.

* **Safety and Consistency**: Raft guarantees consistency by imposing
  restrictions on leader election and log structure. For example, candidate
  nodes must have at least up-to-date logs as the current nodes, and a committed
  log entry must have the same operation and index across all nodes.

**Raft** improves upon **Paxos** by providing a better separation of
responsibilities and a more intuitive approach to managing the distributed
consensus problem. It has become popular for its ease of understanding,
implementation, and use in building distributed systems, although it may have
some trade-offs in terms of performance and message overhead compared to Paxos.

# CRDT's (convergent replicated data types) 

A class of data structures specifically designed for distributed systems to
ensure eventual consistency and conflict-free replication across multiple nodes.
CRDTs enable nodes to independently and concurrently update their local data and
eventually synchronize with each other, resulting in a consistent view of the
shared data. This synchronization can happen in a peer-to-peer or gossip-style
communication, without relying on a central coordinator or consensus algorithm
like Paxos or Raft.

CRDTs guarantee that each replica will converge to the same state

# CALM (consistency as logical monotonicity) 

A principle in distributed systems that provides a way to reason about the
consistency of data and computation in a distributed environment. It is based on
the idea that if the logic of a distributed system can be expressed using
monotonic operations (i.e., functions that only accumulate information and never
contradict or retract previous information), then the system can automatically
ensure eventual consistency.

CALM helps in determining when coordination is necessary between distributed
components and when it can be avoided, leading to more efficient and scalable
solutions. The main idea is that monotonic systems do not need synchronization
protocols to maintain consistency, but non-monotonic systems require some form
of synchronization or coordination to ensure consistency.

In essence, CALM provides a guideline for designing distributed systems with the
following benefits:

* **Scalability**: By minimizing or avoiding coordination, systems can scale
  better as they can handle more nodes and communication requirements.
* **Fault-tolerance**: Monotonic systems tend to be more resilient to failures,
  as they do not require complex synchronization protocols that may be
  susceptible to faults.
* **Eventual consistency**: Monotonic systems can guarantee eventual
  consistency, which means that once all input data is processed, the system
  will eventually reach a consistent state.

Overall, **CALM** provides a framework that helps distributed systems designers
to understand where consistency issues may arise and how to address them
effectively. This leads to creating more reliable, efficient, and scalable
distributed systems.
