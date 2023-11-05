# Materials

* [Actor Model Explained | youtube](https://www.youtube.com/watch?v=ELwEdb_pD0k&t=269s)

# Basic

## Concept

The **Actor model** in computer science is a mathematical framework for concurrent
and distributed computation. It was first introduced by Carl Hewitt, Peter
Bishop, and Richard Steiger in 1973. The model simplifies the design and
implementation of concurrent and distributed systems by providing high-level
abstractions for concurrency, communication, and synchronization.

In the Actor model, an actor is a fundamental unit of computation that
encapsulates its state and behavior. Actors communicate with each other by
exchanging asynchronous messages. Each actor has a unique address and performs
actions in response to the messages it receives. These actions can involve local
processing, creating new actors, or sending messages to other actors.

Key ideas in the Actor model include:

- **Concurrency**: Actors run concurrently and independently, which means that
  they can perform their tasks without waiting for other actors or sharing
  resources, such as memory or computation.
- **Isolation**: Each actor encapsulates its state and behavior, preventing
  other actors from directly accessing or manipulating its internal data. This
  promotes modularity and fault tolerance.
- **Asynchronous communication**: Actors communicate through message passing,
  allowing them to interact without being tightly coupled or blocked by other
  actors' computations. This enables high concurrency in a system with many
  actors.
- **Location transparency**: Actors can communicate and interact regardless of
  their physical location, whether they reside in the same process, across
  processes, or even on different machines. This allows for scalable and
  distributed computing.

The Actor model has been widely adopted in various programming languages and
systems, such as Erlang, Akka (for Scala and Java), Orleans (for .NET), and
Pony. It has been used in building highly concurrent, fault-tolerant, and
scalable systems, such as telecommunications switches, web servers, and
applications for big data processing.

## Actor Model Frameworks

- Erlang:
  - **OTP (Open Telecom Platform)**: A set of libraries and middleware that provide
    the basis for concurrent, fault-tolerant, and distributed systems in Erlang.
    OTP also includes the Erlang/OTP GenServer, an abstraction to write
    actor-based applications.
- Elixir:
  - Elixir is built on top of the Erlang runtime system and takes advantage of
    the Actor model through OTP. It essentially uses the same toolset as Erlang,
    with added syntactic sugar.
- Scala & Java:
  - **Akka**: Akka is a toolkit and runtime for building highly concurrent,
    distributed, and fault-tolerant systems in both Scala and Java. It provides
    abstractions such as Actors, Supervisors, and Cluster support.
  - **Apache Pekko**: Apache Pekko is an open-source framework for building
    applications that are concurrent, distributed, resilient and elastic. Pekko
    uses the Actor Model to provide more intuitive high-level abstractions for
    concurrency. Using these abstractions, Pekko also provides libraries for
    persistence, streams, HTTP, and more. Pekko is a fork of Akka 2.6.x, prior
    to the Akka project's adoption of the Business Source License.
- 4.C# and .NET:
  - **Microsoft Orleans**: A framework for building distributed, high-scale
    applications on the .NET platform. Orleans implements the Actor model and
    provides virtual actors, designed to simplify distributed system design.
