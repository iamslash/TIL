- [Materials](#materials)
- [Basic](#basic)
  - [Concept](#concept)
  - [E-Commerce Akka Example](#e-commerce-akka-example)
  - [E-Commer Akka Example For Scalability](#e-commer-akka-example-for-scalability)
  - [E-Commer Akka Example For Faul-Tollerant](#e-commer-akka-example-for-faul-tollerant)
  - [E-Commer Akka Example For Location Transparency](#e-commer-akka-example-for-location-transparency)
  - [Actor Model Frameworks](#actor-model-frameworks)

------

# Materials

* [Actor Model Explained | youtube](https://www.youtube.com/watch?v=ELwEdb_pD0k&t=269s)

# Basic

## Concept

The **Actor model** in computer science is a mathematical framework for concurrent
and distributed computation. It was first introduced by Carl Hewitt, Peter
Bishop, and Richard Steiger in 1973. The model simplifies the design and
implementation of concurrent and distributed systems by providing high-level
abstractions for concurrency, communication, and synchronization.

The Actor Model is considered mathematical because it is a well-defined, formal model that describes concurrent and distributed computation based on a set of mathematical rules and principles.

In the Actor Model, the basic units of computation are actors. An actor is able
to:

- Receive and process messages from its mailbox.
- Send messages to other actors.
- Create new actors.
- Change its internal state and behavior.

The mathematical aspect of the Actor Model comes from the rigorous definition of
these rules and the way they interact with each other. The following are some of
the key features that contribute to the mathematical nature of the Actor Model:

- **Formal semantics**: The Actor Model has well-defined operational semantics
  that describe precisely how messages are sent, received, and processed by
  actors, as well as how new actors are created.
- **Determinism and non-determinism**: The Actor Model captures the essence of
  non-determinism in concurrent systems. It acknowledges that the order of
  message arrivals and processing is not guaranteed, and provides mathematical
  tools to reason about this inherent non-determinism.
- **Unbounded concurrency**: The Actor Model defines a theoretically unbounded
  number of actors, which allows it to model potentially infinite concurrent
  computations.
- **Location transparency**: The Actor Model includes the concept of location
  transparency, which means that actors can communicate with one another
  independent of their location (local or remote). This feature allows for a
  formal, mathematical treatment of distributed computing.
- **Strong encapsulation**: Actor Model enforces strong encapsulation of state
  management, meaning that an actor's internal state is only affected by the
  messages it processes, and cannot be directly accessed or modified by other
  actors.

These mathematical properties of the Actor Model enable developers to reason
about complex concurrent and distributed systems more effectively. They help
provide guarantees around the behavior of concurrent systems and guide the
design of scalable, fault-tolerant, and resilient applications.

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

## E-Commerce Akka Example 

```java
// Messages
public interface Message {}

public class ProcessOrder implements Message {
    private final Order order;

    public ProcessOrder(Order order) {
        this.order = order;
    }

    public Order getOrder() {
        return order;
    }
}

public class GetPaymentStatus implements Message {
    private final String orderId;

    public GetPaymentStatus(String orderId) {
        this.orderId = orderId;
    }

    public String getOrderId() {
        return orderId;
    }
}

public class UpdateStock implements Message {
    private final Order order;

    public UpdateStock(Order order) {
        this.order = order;
    }

    public Order getOrder() {
        return order;
    }
}

// Actors
public class OrderActor extends AbstractActor {
    private final Map<String, Order> orders = new HashMap<>();

    @Override
    public Receive createReceive() {
        return receiveBuilder()
                .match(ProcessOrder.class, po -> {
                    orders.put(po.getOrder().getId(), po.getOrder());
                    getSender().tell("Order processed successfully", getSelf());
                })
                .build();
    }
}

public class PaymentActor extends AbstractActor {
    @Override
    public Receive createReceive() {
        return receiveBuilder()
                .match(GetPaymentStatus.class, gps -> {
                    String orderId = gps.getOrderId();
                    // On retrieving the payment status for the given order id.
                    getSender().tell("Payment successful for order: " + orderId, getSelf());
                })
                .build();
    }
}

public class InventoryActor extends AbstractActor {
    @Override
    public Receive createReceive() {
        return receiveBuilder()
                .match(UpdateStock.class, us -> {
                    Order order = us.getOrder();
                    // On updating the stock for the given order.
                    getSender().tell("Stock updated for order: " + order.getId(), getSelf());
                })
                .build();
    }
}
```

Consider an e-commerce application that handles customer orders. We will have
three main components.
- **OrderActor**: This actor handles the logic of creating, updating, and
  retrieving orders.
- **PaymentActor**: This actor handles payment processing and transactions.
- **InventoryActor**: This actor handles the logic of updating inventory stock
  and checking item availability.

each component (order, payment, and inventory) of the e-commerce application is
represented by an independent actor that communicates using asynchronous
messages. This level of decoupling and concurrency can greatly improve the
efficiency of the system, especially in high-load situations.

Compared to traditional Spring Boot applications, Actor Model provides better
support for **concurrent** and **distributed systems**. Spring Boot applications
often use shared **mutable state** and **thread synchronization** for concurrent
tasks, which can lead to **complex** and **error-prone code**. On the other
hand, the Actor Model emphasizes **message-passing** and **immutability**, which
helps to develop robust and scalable applications more efficiently.

But we can make order-api, payment-api, inventory-api microservices using spring very easily. 

Here's a quick comparison of the two:

- Ease of development:
  - **Spring Boot**: It is well-known for its ease of development, providing
    many out-of-the-box features, and relying heavily on the dependency
    injection pattern. Developers can quickly set up and deploy microservices
    for order-api, payment-api, and inventory-api using Spring Boot.
  - **Akka**: While Actor Model has a learning curve and requires a different
    mindset, Akka makes it easy to create highly concurrent and fault-tolerant
    applications once you are familiar with the Actor Model.
- Concurrency:
  - **Spring Boot**: Spring Boot can handle concurrent tasks by using `@Async`
    annotations and thread pools. However, managing concurrent state can become
    complex with traditional approaches that rely on shared mutable state and
    synchronization.
  - **Akka**: The Actor Model emphasizes message-passing and immutability,
    making it easier to manage concurrent state. Akka actors can also easily
    scale to a large number of concurrent entities.
- Scalability and resilience:
  - **Spring Boot**: Spring Boot microservices can be individually scaled and
    deployed in a distributed environment. However, fault tolerance and
    self-healing capabilities may require additional configuration and
    consideration.
  - **Akka**: Akka is specifically designed to create highly scalable and
    fault-tolerant applications with built-in support for supervision and
    monitoring of actors. It also supports location transparency, which makes it
    easy to distribute actors across a network.
- Communication:
  - **Spring Boot**: Microservices generally communicate through RESTful APIs, which are simple and widely adopted. However, this design may introduce some latency due to HTTP overhead.
  - **Akka**: Actors communicate using asynchronous messages, which enables highly concurrent communication patterns and efficient resource utilization.

Besides we can make it using order-consumer, payment-consumer,
inventory-consumer microservices with Kafka. 

Using order-consumer, payment-consumer, and inventory-consumer microservices
with Kafka is not an **Actor Model** implementation, but it shares some
similarities with the Actor Model, particularly in terms of message-passing and
concurrency.

The architecture you described involves a set of microservices communicating
asynchronously via Kafka, which is a distributed streaming platform. Kafka, as
the messaging system, takes care of message delivery, ordering, and persistence
between the microservices. The microservices themselves can process messages
concurrently without being blocked by other components.

The Actor Model, on the other hand, is a general mathematical framework for
concurrent and distributed computation where actors are the fundamental units
that communicate using asynchronous message-passing. Akka is a popular
implementation of the Actor Model that makes it easy to build highly concurrent
and distributed systems.

The similarities between the two architectures include:

- **Asynchronous communication**: Both Kafka-based microservices and the Actor Model
  rely on asynchronous message-passing to handle inter-component communication,
  which facilitates concurrent and non-blocking processing.
- **Decoupling of components**: In both cases, the components (microservices or
  actors) operate independently, meaning they can be scaled, modified, and
  deployed without affecting the other parts of the system.
- **Resilience**: Both architectures can be designed to be fault-tolerant and
  resistant to failures in individual components.

However, the differences between Kafka-based microservices and the Actor Model
are as follows:

- **Framework and programming model**: An Actor Model implementation, like Akka,
  provides a specific framework and programming paradigm for building concurrent
  and distributed systems. In contrast, Kafka-based microservices use a more
  general microservices architecture, and the choice of implementation framework
  and concurrency model is left to the developer.
- **Location transparency and distribution**: The Actor Model supports location
  transparency, enabling actors to be automatically distributed across a
  network, whereas in Kafka-based microservices, the responsibility lies with
  the developer to deploy and manage microservices in a distributed environment.
- **Fault tolerance and supervision**: Akka's Actor Model includes built-in
  support for hierarchical supervision and self-healing, while fault tolerance
  in Kafka-based microservices must be designed and implemented by the
  developers.

Using order-consumer, payment-consumer, and inventory-consumer microservices
with Kafka is not an Actor Model implementation, but it shares some similarities
in terms of message-passing and concurrency. Each architecture has its own
advantages and trade-offs, and the choice depends on the desired features,
programming model, and use case.

## E-Commer Akka Example For Scalability

`orderActorRouter` is a kind of a gateway of `orderActors`.

```java
import akka.actor.AbstractActor;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.routing.RoundRobinPool;

public class EcommerceApp {
    public interface Message {}
    
    // Message classes here (ProcessOrder, GetPaymentStatus, UpdateStock) ...

    public class OrderActor extends AbstractActor {
        // OrderActor implementation here ...
    }

    public class PaymentActor extends AbstractActor {
        // PaymentActor implementation here ...
    }

    public class InventoryActor extends AbstractActor {
        // InventoryActor implementation here ...
    }

    public static void main(String[] args) {
        ActorSystem system = ActorSystem.create("ecommerce");
        
        int numOrderActors = 5;
        int numPaymentActors = 3;
        int numInventoryActors = 4;
        
        // Creating actor pools using routers and round-robin strategy
        ActorRef orderActorRouter = system.actorOf(
            new RoundRobinPool(numOrderActors).props(Props.create(OrderActor.class)), "orderActorRouter");
        ActorRef paymentActorRouter = system.actorOf(
            new RoundRobinPool(numPaymentActors).props(Props.create(PaymentActor.class)), "paymentActorRouter");
        ActorRef inventoryActorRouter = system.actorOf(
            new RoundRobinPool(numInventoryActors).props(Props.create(InventoryActor.class)), "inventoryActorRouter");
        
        // Sending messages to routers, which will distribute the messages to the actor pool
        // The following can be adapted to your specific use case (e.g., a loop to simulate multiple requests)
        Order testOrder = new Order("1", "item1", 1);
        orderActorRouter.tell(new ProcessOrder(testOrder), ActorRef.noSender());
        paymentActorRouter.tell(new GetPaymentStatus(testOrder.getId()), ActorRef.noSender());
        inventoryActorRouter.tell(new UpdateStock(testOrder), ActorRef.noSender());
        
        // Shutting down the actor system and releasing resources
        system.terminate();
    }
}
```

## E-Commer Akka Example For Faul-Tollerant

```java
import akka.actor.AbstractActor;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.OneForOneStrategy;
import akka.actor.Props;
import akka.actor.SupervisorStrategy;
import akka.japi.pf.DeciderBuilder;
import akka.routing.RoundRobinPool;

public class EcommerceApp {
    public interface Message {}
    
    // Message classes here (ProcessOrder, GetPaymentStatus, UpdateStock) ...

    public class SimulateFailure implements Message {}

    public class OrderActor extends AbstractActor {
        // OrderActor implementation here ...
        
        // Adding a behavior to simulate a failure
        @Override
        public Receive createReceive() {
            return receiveBuilder()
                    .match(ProcessOrder.class, po -> {
                        /* ... */
                    })
                    .match(SimulateFailure.class, sf -> {
                        throw new RuntimeException("Simulated failure in OrderActor");
                    })
                    .build();
        }
    }

    public class PaymentActor extends AbstractActor {
        // PaymentActor implementation here ...
    }

    public class InventoryActor extends AbstractActor {
        // InventoryActor implementation here ...
    }

    public class SupervisorActor extends AbstractActor {
        private final ActorRef orderActorRouter;

        public SupervisorActor() {
            orderActorRouter = getContext().actorOf(
                    new RoundRobinPool(5).props(Props.create(OrderActor.class)), "orderActorRouter");
        }

        @Override
        public SupervisorStrategy supervisorStrategy() {
            return new OneForOneStrategy(false, DeciderBuilder
                    .match(RuntimeException.class, e -> SupervisorStrategy.restart())
                    .matchAny(e -> SupervisorStrategy.escalate())
                    .build());
        }

        @Override
        public Receive createReceive() {
            return receiveBuilder()
                    .matchAny(msg -> {
                        orderActorRouter.forward(msg, getContext());
                    })
                    .build();
        }
    }

    public static void main(String[] args) {
        ActorSystem system = ActorSystem.create("ecommerce");

        ActorRef supervisorActor = system.actorOf(Props.create(SupervisorActor.class), "supervisorActor");
        ActorRef paymentActorRouter = system.actorOf(
            new RoundRobinPool(3).props(Props.create(PaymentActor.class)), "paymentActorRouter");
        ActorRef inventoryActorRouter = system.actorOf(
            new RoundRobinPool(4).props(Props.create(InventoryActor.class)), "inventoryActorRouter");
        
        // Sending messages to supervised OrderActor
        Order testOrder1 = new Order("1", "item1", 1);
        supervisorActor.tell(new ProcessOrder(testOrder1), ActorRef.noSender());

        // Simulating a failure in OrderActor
        supervisorActor.tell(new SimulateFailure(), ActorRef.noSender());

        // Sending another message to see if the failed OrderActor recovers
        Order testOrder2 = new Order("2", "item2", 2);
        supervisorActor.tell(new ProcessOrder(testOrder2), ActorRef.noSender());
        
        // Sending messages to the other routers
        paymentActorRouter.tell(new GetPaymentStatus(testOrder1.getId()), ActorRef.noSender());
        inventoryActorRouter.tell(new UpdateStock(testOrder1), ActorRef.noSender());
        
        system.terminate();
    }
}
```

In this example, we introduced the `SupervisorActor` class, which acts as a
supervisor responsible for handling faults in the `OrderActor`. Each `OrderActor`
instance is supervised by the `SupervisorActor`, which can decide the course of
action upon failure (e.g., restart, stop, escalate the problem). In the provided
example, we are using a `OneForOneStrategy` to restart the failed `OrderActor`
whenever a `RuntimeException` occurs.

The `SimulateFailure` message is used to simulate an intentional failure in an
OrderActor. When an `OrderActor` receives a `SimulateFailure` message, it throws
a `RuntimeException`. The `SupervisorActor` detects the failure and restarts the
affected `OrderActor`, allowing the actor system to continue processing incoming
messages.

## E-Commer Akka Example For Location Transparency

Taking advantage of the location transparency provided by Akka Clustering. As a
reminder, it's important to note that this example requires a proper clustering
setup, including multiple nodes and proper configuration.

First, add the `akka-cluster` and `akka-cluster-tools` dependencies to your project.
For Maven, add the following to your pom.xml:

```xml
<!-- pom.xml -->
<dependencies>
   <!-- ... -->

   <dependency>
     <groupId>com.typesafe.akka</groupId>
     <artifactId>akka-cluster_2.11</artifactId>
     <version>2.5.32</version>
   </dependency>
   <dependency>
     <groupId>com.typesafe.akka</groupId>
     <artifactId>akka-cluster-tools_2.11</artifactId>
     <version>2.5.32</version>
   </dependency>
</dependencies>
```

Add the following configuration for Akka clustering to the application.conf file:

```conf
# application.conf
akka {
  loglevel = "DEBUG"

  actor {
    provider = "cluster"
  }
  
  remote {
    artery {
      enabled = on
      transport = tcp
      canonical.hostname = "127.0.0.1"
      // Give different port numbers for each node, e.g., 2551, 2552, 2553
      canonical.port = 2551
    }
  }

  cluster {
    seed-nodes = [
      "akka://ecommerce@127.0.0.1:2551"]
  }
}
```

Now, here's the modified e-commerce Akka Java code with location transparency
using Akka Clustering:

```java
import akka.actor.AbstractActor;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.cluster.Cluster;
import akka.cluster.routing.ClusterRouterPool;
import akka.cluster.routing.ClusterRouterPoolSettings;
import akka.routing.RoundRobinPool;

public class EcommerceApp {
    public interface Message {}

    // Message classes here (ProcessOrder, GetPaymentStatus, UpdateStock) ...

    public class OrderActor extends AbstractActor {
        // OrderActor implementation here ...
    }

    public class PaymentActor extends AbstractActor {
        // PaymentActor implementation here ...
    }

    public class InventoryActor extends AbstractActor {
        // InventoryActor implementation here ...
    }

    public static void main(String[] args) {
        // Command line argument to specify node port (e.g., -DPORT=2552)
        String port = System.getProperty("PORT", "2551");
        System.setProperty("akka.remote.artery.canonical.port", port);
        ActorSystem system = ActorSystem.create("ecommerce");

        // Get the actor cluster for current actor system
        Cluster cluster = Cluster.get(system);

        // Create cluster-aware routers for each actor type
        int totalInstances = 100;
        int maxInstancesPerNode = 3;
        boolean allowLocalRoutees = true;

        ClusterRouterPoolSettings orderActorRouterSettings = new ClusterRouterPoolSettings(
            totalInstances, maxInstancesPerNode, allowLocalRoutees, null);

        ActorRef orderActorRouter = system.actorOf(
            new ClusterRouterPool(new RoundRobinPool(0), orderActorRouterSettings).props(Props.create(OrderActor.class)),
            "orderActorRouter");

        ClusterRouterPoolSettings paymentActorRouterSettings = new ClusterRouterPoolSettings(
            totalInstances, maxInstancesPerNode, allowLocalRoutees, null);

        ActorRef paymentActorRouter = system.actorOf(
            new ClusterRouterPool(new RoundRobinPool(0), paymentActorRouterSettings).props(Props.create(PaymentActor.class)),
            "paymentActorRouter");

        ClusterRouterPoolSettings inventoryActorRouterSettings = new ClusterRouterPoolSettings(
            totalInstances, maxInstancesPerNode, allowLocalRoutees, null);

        ActorRef inventoryActorRouter = system.actorOf(
            new ClusterRouterPool(new RoundRobinPool(0), inventoryActorRouterSettings).props(Props.create(InventoryActor.class)),
            "inventoryActorRouter");
        
        // Send messages to the cluster-aware actor routers
        // The following can be adapted to your specific use case (e.g., a loop to simulate multiple requests)
        Order testOrder = new Order("1", "item1", 1);
        orderActorRouter.tell(new ProcessOrder(testOrder), ActorRef.noSender());
        paymentActorRouter.tell(new GetPaymentStatus(testOrder.getId()), ActorRef.noSender());
        inventoryActorRouter.tell(new UpdateStock(testOrder), ActorRef.noSender());
    }
}
```

In this example, we created a cluster-aware router for each actor type
(orderActor, paymentActor, and inventoryActor) using `ClusterRouterPool`, which
takes care of automatically routing messages to actor instances across nodes in
the cluster.

This approach leverages Akka's location transparency, allowing the application
to scale out by distributing the actors across different nodes in the cluster.
This increases the system's ability to handle higher workloads and provides
fault tolerance by reducing single points of failure.

To test this setup, you can start multiple instances of the application (nodes)
with the same code on different ports (e.g., 2551, 2552, 2553) by providing
different port numbers via the PORT system property.

Keep in mind that this example is just a starting point to demonstrate the
concept of location transparency. In a real-world scenario, you need to
fine-tune your configuration, define the proper roles for cluster nodes to host
specific types of actors, and implement the actual message processing logic in
the actors.

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
