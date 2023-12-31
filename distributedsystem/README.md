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

- [Consensus in Distributed System](https://medium.com/@sourabhatta1819/consensus-in-distributed-system-ac79f8ba2b8c)

분산 컴퓨팅에서 합의 문제는 여러 프로세스 또는 노드가 개별 입력에 기반하여 공통
값을 합의해야 하는 기본적인 도전 과제입니다. 이 문제는 분산 트랜잭션, 분산
데이터베이스, 리더 선출, 상태 머신 복제 등의 실용적인 시나리오에서 종종
발생하며, 분산 시스템 내 노드들이 전체 일관성과 올바른 기능을 유지하기 위해
값이나 행동에 대해 합의해야 하는 경우가 대표적입니다.

합의 알고리즘은 다음 중요한 속성을 만족하면서 합의 문제를 해결하는 데 목표를
두고 있습니다.

* **합의(Agreement)**: 모든 비결함 노드가 동일한 값을 합의해야 합니다.
* **유효성(Validity)**: 합의된 값은 노드 중 하나의 입력 값이어야 합니다.
* **종료(Termination)**: 모든 비결함 노드는 결국 값을 결정해야 합니다.
* **무결성(Integrity)**: 노드는 최대한 한 번만 값을 결정해야 하며, 값에 결정하는
  경우 처음에 제안했거나 다른 노드로부터 받은 값이어야 합니다.

결함이 있는 노드가 존재하는 경우 합의 문제는 더욱 어려워집니다. 이러한 노드는
실패, 충돌 또는 비잔틴 행동(노드가 예측할 수 없거나 모순되거나 악의적인 행동을
할 수 있는 경우)을 보일 수 있습니다. **합의 알고리즘**은 원하는 속성을 보장하는
동시에 일정 수의 결함 노드도 용인해야 합니다.

일부 잘 알려진 합의 알고리즘 및 프로토콜은 다음과 같습니다.

* **Paxos**: 비잔틴 결함(노드 충돌과 같은 사례)을 용인하며, 부분적으로 동기화된
  시스템에서 안전성(합의 및 유효성)과 활성성(종료)을 보장하는 합의
  알고리즘입니다.
* **Raft**: 가독성 및 이해성을 위해 특별히 설계된 합의 알고리즘으로, Paxos와
  유사한 보장을 제공하지만 더 간결하고 직관적인 접근 방식과 구현을 제공합니다.
* **비잔틴 장애 허용 프로토콜(Byzantine fault-tolerant, BFT, protocols)**:
  노드가 임의로 혹은 악의적인 행동을 취할 수 있는 환경에서 합의 보장을 제공하는
  알고리즘으로 Practical Byzantine Fault Tolerance(PBFT) 및 Stellar Consensus
  Protocol(SCP)이 있습니다.
* **작업 증명(Proof-of-Work, PoW), 지분 증명(Proof-of-Stake, PoS)**: 비트코인과
  이더리움과 같은 암호화폐 시스템에서 분산 성격의 글로벌 원장 상태에 합의를
  이루기 위해 사용되는 합의 메커니즘입니다.

합의 문제를 해결하는 것은 분산 시스템의 **정확성**, **일관성**, **신뢰성**을
보장하는 데 필수적입니다. 이를 통해 네트워크 대기 시간, 노드 실패 및 비동기 등이
내재된 도전에도 불구하고 효과적으로 작업을 조율할 수 있습니다.

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

Paxos 프로토콜은 분산 시스템에서 사용되는 합의 알고리즘으로, 노드 실패, 충돌
또는 네트워크 지연이 발생할 경우에도 노드 집합이 단일 값을 동의할 수 있는지
확인합니다. Paxos는 과반수의 노드가 정상적으로 작동하고 통신하는 한 합의가
이루어지도록 내결함성을 가지도록 설계되었습니다. 분산 데이터베이스, 상태 기계
복제 및 분산 트랜잭션 관리와 같이 여러 노드가 일관된 값이나 작업에 동의해야 하는
응용 프로그램에서 주로 사용됩니다.

Paxos의 핵심 개념은 노드의 과반수가 해당 값을 수용한 경우 해당 값이 합의 된
것으로 간주된다는 것입니다. 제안 번호 순서대로 값을 선택함으로써 프로토콜은
고유한 제안자에 의해 값이 선택되고, 이 결정된 값이 최종적으로 동의된 값이 될
것임을 보장합니다.

Paxos 프로토콜은 세 가지 주요 역할을 포함합니다:

* **Proposers**: 합의할 값에 대한 제안을 발표하는 노드입니다.
* **Acceptors**: 제안을 투표하며, Paxos 규칙에 따라 제안을 수락하거나 거부하는 노드입니다.
* **Learners**: 결정된 값을 알게 되어 적절한 조치를 취하는 노드입니다.

Paxos는 두 가지 주요 단계로 작동합니다:

* **Prepare Phase**: 
  * 제안자가 제안 번호 n을 선택하고 "준비 요청" 메시지를 과반수의 수락자에게
    보냅니다.
  * 수락자가 준비 요청을 받으면 제안 번호 n이 이전에 본 제안보다 높은지
    확인합니다. 만약 그렇다면, 수락자는 메시지를 승인하고, n보다 낮은 번호의
    제안을 받지 않겠다는 약속을 하고, 이전에 수락한 최고 제안 (있는 경우)을
    제안자에게 다시 보냅니다.
* **Accept Phase**: 
  * 제안자가 수락자의 과반수로부터 응답을 받습니다. 그 후 응답으로부터 가장 높은
    번호의 제안과 연관된 값을 선택하거나 (이전에 받아들여진 제안이 없는 경우 새
    값을 사용)하고, (n, 값) 쌍으로 된 "수락 요청"을 과반수의 수락자에게
    보냅니다.
  * 수락자가 수락 요청을 받고 n 이하의 번호를 가진 제안을 무시하기로 한 약속이
    없으면, 제안을 수락하고 "승인됨" 메시지를 제안자와 학습자에게 보냅니다.

**제안자**와 **학습자**가 **수락자**의 과반수로부터 "승인됨" 메시지를 받으면,
값은 결정된 것으로 간주되며 합의가 이루어집니다.

**Paxos**는 모든 조건에서 안전성(다른 값에 동의 할 수 없다는 것을 보장)을
보장하고 실패 또는 비동기가 없는 경우 활성성(결국 값이 합의된다는 것을 보장)을
보장합니다. 그러나 이 프로토콜은 이해하고 구현하기 복잡할 수 있습니다.
**Multi-Paxos**와 같은 변형 및 **Raft**와 같은 다른 합의 알고리즘이 기본 Paxos
프로토콜의 일부 복잡성과 특정 측면을 개선하기 위해 개발되었습니다.

Paxos 알고리즘이 사용되는 주요 예제와 응용 프로그램은 다음과 같습니다:

- 분산 데이터베이스: Paxos는 다양한 분산 데이터베이스 시스템에서 사용되어 노드
  간의 일관된 상태를 유지합니다. 일례로, Google의 Spanner 데이터베이스 시스템은
  Paxos 알고리즘을 사용하여 데이터베이스의 일관성을 유지하고 범위 데이터를
  전역으로 분산합니다.
- 분산 트랜잭션 관리: Paxos는 여러 노드에서 트랜잭션의 순서와 일관성을
  합의하도록 지원하는 분산 트랜잭션 관리 시스템에서 사용됩니다.
- 상태 기계 복제: 상태 기계 복제는 분산 시스템 내에서 비동기적 상태 변경을
  처리하기 위한 핵심 기술입니다. Paxos는 상태 기계 간의 순서와 일관성을 유지하는
  데 사용됩니다.
- 종료 감지: Paxos는 분산 시스템에서 프로세스의 종료를 검출하고 모니터링하는 데
  사용되는 알고리즘이기도 합니다. 이는 천명치 않은 프로세스의 종료에 의해
  시스템의 정확한 작동이 보장되어야 하는 경우 유용합니다.
- 리더 선출: Paxos는 리더 선출 메커니즘에서 중요한 역할을 합니다. 분산
  시스템에서 일반적으로 리더 노드가 다른 노드들에게 결정을 내리거나 명령을
  내리고, Paxos는 여러 노드에서 리더를 동의하여 선출하는 데 사용됩니다.

기술적으로 말하면, Paxos는 수많은 분산 시스템에서 사용되지만 일부 복잡성과
구현의 어려움 때문에 사용되는 경우 눈에 띄지 않을 수 있습니다. 많은 경우 기본
Paxos 알고리즘이 등장하여 Multi-Paxos, EPaxos, Fast Paxos 등과 같은 개량된
버전이 개발되었습니다.

```java
// Proposer, Acceptor, Learner Interfaces
public interface Proposer {
    void prepare(int proposalNumber);
    void receivePrepareResponse(Promise promise);
    void sendAcceptRequest();
}

public interface Acceptor {
    Promise prepare(int proposalNumber);
    boolean accept(Proposal proposal);
}

public interface Learner {
    void learn(Proposal proposal);
}

// Proposer, Acceptor, Learner Implementation
public class BasicProposer implements Proposer {
    // Implementation details...
}

public class BasicAcceptor implements Acceptor {
    // Implementation details...
}

public class BasicLearner implements Learner {
    // Implementation details...
}

// Entry
public class PaxosExample {
    public static void main(String[] args) {
        // Create instances of Proposer, Acceptor, and Learner
        Proposer proposer = new BasicProposer();
        Acceptor acceptor = new BasicAcceptor();
        Learner learner = new BasicLearner();

        // Run the Prepare phase
        proposer.prepare(1);

        // Receive a Prepare response (simulate with a Promise object)
        Promise promise = acceptor.prepare(1);
        proposer.receivePrepareResponse(promise);

        // Send an Accept request
        proposer.sendAcceptRequest();

        // Simulate an acceptor accepting the proposal and a learner learning it
        Proposal proposal = new Proposal(1, "Value");
        boolean accepted = acceptor.accept(proposal);
        if (accepted) {
            learner.learn(proposal);
        }

        // Test Paxos
        System.out.println("Paxos completed");
    }
}
```

Paxos와 Raft는 분산 시스템에서 합의를 이루는 데 사용되는 두 가지 알고리즘입니다.
두 알고리즘의 주요 차이점은 다음과 같습니다:

- 이해하기 쉬움: Raft는 Paxos보다 이해하기 쉽게 설계되었습니다. Paxos가 설명서와
  학습 자료가 거의 없고 이해하기 어렵다면, Raft는 쉽게 접근할 수 있는 문서와
  예제로 구성되어 있습니다. 이로 인해 Raft를 사용하여 더 빠르게 프로젝트를
  작성하고 유지 관리할 수 있습니다.
- 구조: Paxos는 일련의 독립적인 프로토콜로 구성되어 있어 구현이 복잡할 수
  있습니다. 반면 Raft는 모듈화된 설계가 적용되어 있어 구현 및 학습에 장벽이
  적습니다.
- 리더 선출: Paxos는 리더 선출이 암시적으로 이루어지는 반면, Raft는 명시적으로
  리더 선출 프로세스를 가지고 있습니다. 이로써 Raft에서는 중앙 집중적인 리더
  노드가 시스템에서 모든 판단을 내리며, 이는 제어 및 디버깅을 용이하게 합니다.
- 판결 메커니즘: Raft에서 투표는 일련의 연속적인 항목으로 구성됩니다. 이는 모든
  노드를 연속된 항목으로 동기화할 수 있는 알고리즘을 사용합니다. 반면
  Paxos에서는 논리적 순서에 따라 투표를 선택할 수 있고, 독립된 항목을 기반으로
  작동합니다.
- 안정성 및 최적화: Paxos에서는 각 단계에 대해 별도의 알고리즘을 사용하여
  최적화를 수행할 수 있습니다. 반면에 Raft에서는 일반적으로 알고리즘 전체를
  최적화하고 개선합니다. 이로 인해 Paxos의 성능은 경우에 따라 Raft보다 더 높을
  수 있습니다. 그러나 Raft의 단순성으로 인해 실제 구현에서 종종 더 나은 결과를
  얻을 수 있습니다.

결론적으로, 두 알고리즘의 목적이 동일하고 한 알고리즘이 다른 것보다 좋은 성능을
내는 것은 상황에 따라 다릅니다. 그러나 Raft의 이해하기 쉬운 구조와 명시적인 리더
선출, 구현의 효과성으로 인해 많은 신규 프로젝트에서 선호되는 경향이 있습니다.

# Raft

> * [Raft 분산 합의 알고리즘과 Python에서의 활용 @ NAVERD2](https://d2.naver.com/helloworld/5663184)
>  * [The Raft Consensus Algorithm](https://raft.github.io/)
>  * [raft scope animation](https://raft.github.io/raftscope/index.html)
>  * [pyraft](https://github.com/lynix94/pyraft)
> * [Raft Consensus Algorithm](https://swalloow.github.io/raft-consensus/)

---

Raft와 Gossip Protocol은 분산 시스템에서 사용되는 서로 다른 유형의
알고리즘입니다. 어떤 알고리즘이 더 좋다고 단정지을 수는 없고, 시스템의 요구
사항에 따라 적절한 알고리즘을 선택해야 합니다.

Raft는 분산 시스템에서 독립적인 서버들 사이에 합의를 이루고 중앙 서버가 없을 때
데이터의 일관성을 유지하기 위한 알고리즘 입니다. Raft는 강력한 일관성(Strong
consistency)을 제공하며, 높은 가용성 및 안정성을 지원할 수 있습니다.

Gossip Protocol은 확장성이 뛰어나며 내결함성이 강한 분산 시스템에서의 메시지
전송을 위한 프로토콜입니다. 이 프로토콜은 노드 간 원활한 정보 공유 및 가용성을
제공하지만, 최종 일관성(Eventual consistency)을 제공할 뿐 강력한 일관성은
보장하지 않습니다.

따라서, 시스템이 강력한 일관성을 필요로 하면 Raft를 사용하는 것이 더 좋을 수
있고, 확장성과 내결함성이 더 중요한 요구 사항일 때는 Gossip Protocol이 적합할 수
있습니다. 결정하는 데 있어 사용 사례와 요구 사항이 주요한 역할을 합니다.

다음은 Raft 를 구현한 java code 이다.

```java
// NodeRole.java
public enum NodeRole {
  FOLLOWER,
  CANDIDATE,
  LEADER
}

// RaftNode.java
import java.util.concurrent.atomic.AtomicInteger;

public class RaftNode {
  private String id;
  private NodeRole role;
  private int currentTerm;
  private AtomicInteger votesReceived;

  public RaftNode(String id) {
    this.id = id;
    this.role = NodeRole.FOLLOWER;
    this.currentTerm = 0;
    this.votesReceived = new AtomicInteger();
  }

  public String getId() {
    return id;
  }

  public NodeRole getRole() {
    return role;
  }

  public void setRole(NodeRole role) {
    System.out.println("Node " + id + " changed role from " + this.role + " to " + role);
    this.role = role;
  }

  public int getCurrentTerm() {
    return currentTerm;
  }

  public void setCurrentTerm(int term) {
    System.out.println("Node " + id + " changed term from " + this.currentTerm + " to " + term);
    this.currentTerm = term;
    this.votesReceived.set(0);
  }

  public int getVotesReceived() {
    return votesReceived.get();
  }

  public int incrementVotesReceived() {
    return votesReceived.incrementAndGet();
  }
}

// Election.java
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Election {
  private final List<RaftNode> nodes;

  public Election(int numNodes) {
    nodes = new ArrayList<>(numNodes);
    for (int i = 1; i <= numNodes; i++) {
      nodes.add(new RaftNode("Node" + i));
    }
  }

  public List<RaftNode> getNodes() {
    return nodes;
  }

  public void simulateElection() {
    // 모든 노드를 Candidate로 설정하고 투표를 시작합니다.
    System.out.println("Starting an election...");
    nodes.forEach(node -> node.setRole(NodeRole.CANDIDATE));
    
    // 각 Candidate는 자기 자신에게 투표합니다.
    nodes.forEach(RaftNode::incrementVotesReceived);

    // 각 Follower는 임의로 선출된 후보에게 투표합니다.
    nodes.stream()
        .filter(node -> node.getRole() == NodeRole.FOLLOWER)
        .forEach(follower -> {
          int randomCandidateIndex = new Random().nextInt(nodes.size());
          RaftNode randomCandidate = nodes.get(randomCandidateIndex);
          randomCandidate.incrementVotesReceived();
        });

    // 득표수가 과반수를 넘은 노드 중 하나를 Leader로 선출합니다.
    List<RaftNode> leaders = nodes.stream()
            .filter(node -> node.getVotesReceived() > nodes.size() / 2)
            .collect(Collectors.toList());
    if (!leaders.isEmpty()) {
      RaftNode newLeader = leaders.get(new Random().nextInt(leaders.size()));
      newLeader.setRole(NodeRole.LEADER);
    } else {
      System.out.println("No leader selected. Trying again...");
      simulateElection();
    }
  }
}

// Main.java
public class Main {
  public static void main(String[] args) {
    // 클러스터의 노드 수 설정
    int numNodes = 5;

    // 선택하려는 클러스터의 노드 수로 새 선거 생성
    Election election = new Election(numNodes);

    // 선거 시뮬레이션 실행
    election.simulateElection();

    // 결과 출력
    System.out.println("\nElection results:");
    for (int i = 1; i <= numNodes; i++) {
      RaftNode node = election.getNodes().get(i - 1);
      System.out.println("Node " + i + ": " + node.getRole());
    }
  }
}
```

Raft와 Gossip 알고리즘이 일관성을 관리하는 방식은 매우 다릅니다. Raft는 높은
일관성(strong consistency)을 제공하는 반면 Gossip 알고리즘은 느슨한
일관성(eventual consistency)을 목표로 합니다.

Raft가 높은 일관성을 보장하기 위한 몇 가지 중요한 방법은 다음과 같습니다.

- **리더 선출 (Leader Election)**: Raft 알고리즘은 클러스터의 모든 노드 간에 
  **단일 리더(single leader)**을 선출하여 의사결정의 중앙 집중점이 돼 쓰기와 읽기
  요청을 처리합니다. 이를 통해 쓰기 요청이 순차적으로 처리됩니다. 리더가
  실패하면 다른 노드가 즉시 리더 역할을 가져옵니다.
- **로그 복제 (Log Replication)**: 리더 노드는 모든 합의된 동작을 클러스터 내 다른
  노드에 복제하는 로그를 유지합니다. 로그 항목은 클러스터 노드들과 일관된
  방식으로 적용되므로 각 노드의 상태가 개별적으로 일관성있게 동기화되어 있다고
  가정할 수 있습니다.
- **데이터 변경에 대한 합의 (Consensus over Data Mutations)**: 데이터 변경 요청이
  들어오면 로그 항목을 리더 노드에 추가한 다음, 리더 노드는 해당 로그 항목을
  클러스터의 Follower 노드에 전파합니다. 리더 노드는 과반수의 노드가 로그 항목을
  커밋했을 경우에만 해당 작업이 성공했다고 가정합니다. 이는 일관성있게
  업데이트를 적용할 때 쿼럼(quorum)이 필요함을 보장합니다.

반면 Gossip 알고리즘은 노드가 상태 정보를 주기적으로 무작위로 선택한 다른 노드와
교환하는 방식으로 데이터를 동기화합니다. 이 과정은 특정한 순서가 없으며 특정
시점이되면 모든 노드가 동일한 상태를 갖게 됩니다. 하지만 동시에 여러 노드에서
발생한 업데이트의 경우 일시적인 충돌이 발생할 수 있고, 일관성이 지연되기 때문에
최종적인 일관성(단시간 내 일관성을 보장하지 않음)을 제공합니다.

요약하면, Raft는 데이터 변경에 대한 합의과정, 로그 복제, 그리고 리더 선출을 통해
strong consistency를 보장합니다. 반면 Gossip 알고리즘은 최종적인 일관성(eventual
consistency)을 목표로 합니다.

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
