- [Abstract](#abstract)
- [Materials](#materials)
- [2 Phase Commit](#2-phase-commit)
  - [JAVA Example](#java-example)
- [TCC (Try-Confirm/Cancel)](#tcc-try-confirmcancel)
  - [Sequences](#sequences)
  - [Exceptions](#exceptions)
  - [Summary](#summary)
  - [JAVA Example](#java-example-1)
- [2PC vs TC/C](#2pc-vs-tcc)
- [SAGA](#saga)
  - [SAGA Overview](#saga-overview)
  - [Choreography SAGA](#choreography-saga)
    - [Java Example](#java-example-2)
  - [Orchestration SAGA](#orchestration-saga)
    - [Java Example](#java-example-3)
- [TC/C vs Sagas](#tcc-vs-sagas)
- [Conclusion](#conclusion)

-----

# Abstract

Global transaction 은 local transaction 으로 나누어 진다. 이렇게 local transaction 으로 나누어진 transaction 들의 모음을 distributed transaction 이라고 한다. 

Distributed Transaction 은 다음과 같은 종류가 있다.

* low level (Storage Level)
  * **[2 Phase Commit](/distributedsystem/README.md#2-phase-commit)**
* high level (Application Level)
  * **TC/C (Try Confirm / Cancel)**
  * SAGAS
    * **Choreography SAGAS**
    * **Orchestration SAGAS**

low level means the storage or the driver should support 2 Phase Commit. for
example [MySQL](/mysql/README.md) supports [XA](/mysql/README.md#xa).

high level means the application should support those.

**Choreography** is distributed decision making and **Orchestration** is
centralized decision making. So **Orchestration** has a SPOF problem.

# Materials

* [Distributed Transactions in Microservices with Kafka Streams and Spring Boot](https://piotrminkowski.com/2022/01/24/distributed-transactions-in-microservices-with-kafka-streams-and-spring-boot/)
  * [src](https://github.com/piomin/sample-spring-kafka-microservices) 
* [SAGAS](https://www.cs.cornell.edu/andru/cs711/2002fa/reading/sagas.pdf)
* [Eventuate Tram Sagas](https://eventuate.io/docs/manual/eventuate-tram/latest/getting-started-eventuate-tram-sagas.html)
  * [Managing data consistency in a microservice architecture using Sagas](https://eventuate.io/presentations.html)
  * [Choreography-based sagas example @ github](https://github.com/eventuate-tram/eventuate-tram-examples-customers-and-orders)
  * [orchestration-based-sagas example @ github](https://github.com/eventuate-tram/eventuate-tram-sagas-examples-customers-and-orders)
* [REST 기반의 간단한 분산 트랜잭션 구현 – 1편 TCC 개관](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-1%ED%8E%B8/)
  * [src](https://github.com/YooYoungmo/article-tcc)
  * [REST 기반의 간단한 분산 트랜잭션 구현 - 2편 TCC Cancel, Timeout](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-2%ED%8E%B8-tcc-cancel-timeout/)
  * [REST 기반의 간단한 분산 트랜잭션 구현 - 3편 TCC Confirm(Eventual Consistency)](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-3%ED%8E%B8-tcc-confirmeventual-consistency/)
  * [REST 기반의 간단한 분산 트랜잭션 구현 - 4편 REST Retry](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-4%ED%8E%B8-rest-retry/)
* [대용량 환경에서 그럭저럭 돌아가는 서비스 만들기](https://www.popit.kr/%EB%8C%80%EC%9A%A9%EB%9F%89-%ED%99%98%EA%B2%BD%EC%97%90%EC%84%9C-%EA%B7%B8%EB%9F%AD%EC%A0%80%EB%9F%AD-%EB%8F%8C%EC%95%84%EA%B0%80%EB%8A%94-%EC%84%9C%EB%B9%84%EC%8A%A4-%EB%A7%8C%EB%93%A4%EA%B8%B0/)
* [내 멋대로 구현한 이벤트 드리븐](https://www.popit.kr/%EB%82%B4-%EB%A9%8B%EB%8C%80%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%9C-%EC%9D%B4%EB%B2%A4%ED%8A%B8-%EB%93%9C%EB%A6%AC%EB%B8%90/)
* [마이크로 서비스에서 분산 트랜잭션](https://medium.com/@giljae/%EB%A7%88%EC%9D%B4%ED%81%AC%EB%A1%9C-%EC%84%9C%EB%B9%84%EC%8A%A4%EC%97%90%EC%84%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-347af5136c87)

# 2 Phase Commit

* [Understanding Two-Phase Commit | baeldung](https://www.baeldung.com/cs/saga-pattern-microservices)
  * [A Guide to Transactions Across Microservices | baeldung](https://www.baeldung.com/transactions-across-microservices)

----

다음의 그림과 같이 coordinator 가 여러 microservice 들에게 모두 commit 해도
되는지 물어보고 결정하는 방법이다.

![](img/2phasecommit_prepare.png)

![](img/2phasecommit_commit.png)

다음과 같이 local transaction 을 2 개의 단계로 구분하여 처리한다.

* Prepare Phase
  * commit 해도 되요?
* Commit Phase
  * commit 해 주세요.

Coordinator 가 global transaction, local transaction 의 상태를 저장해야 한다. 각 단계에서 문제가 없다면 모두 commit 한다. 각 단계에서 문제가 발생한다면 전체 transaction 을 rollback 한다.

2 Phase Commit 의 단점은 다음과 같다.

* Coordinator 가 SPOF (Single Point Of Failure) 이다.
* 가장 느린 microservice 에게 bottle neck 이 있다.
* coordinator 를 중심으로 통신이 많다. scalability, performance issue 가 있다.
* NoSQL 은 ACID compliant transaction 을 지원하지 않는다. NoSQL 은 사용이 어렵다. [MySQL](/mysql/README.md) 은 2 phase commit 을 위해 [XA](/mysql/README.md#xa) 를 지원한다.

## JAVA Example

Java에서 2PC를 직접 구현하는 것은 복잡할 수 있으며, 대부분 JTA(Java Transaction API)와 같은 트랜잭션 매니저를 사용하여 처리합니다. 여기서는 JTA를 사용한 간단한 예제를 제공하겠습니다.

JTA를 사용하면 애플리케이션 서버나 트랜잭션 매니저가 2PC 프로토콜의 복잡성을 추상화하고 처리합니다. 아래 예제는 두 개의 데이터베이스 리소스를 사용하는 트랜잭션을 어떻게 관리할 수 있는지 보여줍니다. 이 예제는 JTA를 지원하는 환경에서 실행되어야 하며, 이를 위해 Atomikos와 같은 스탠드얼론 트랜잭션 매니저를 사용하거나, JTA를 지원하는 애플리케이션 서버(예: WildFly, GlassFish)에서 실행할 수 있습니다.

Atomikos를 사용하는 예제 설정입니다. pom.xml에 다음 의존성을 추가합니다:

```xml
<dependencies>
    <!-- Atomikos dependencies -->
    <dependency>
        <groupId>com.atomikos</groupId>
        <artifactId>transactions-jta</artifactId>
        <version>5.0.8</version>
    </dependency>
    <dependency>
        <groupId>com.atomikos</groupId>
        <artifactId>transactions-jdbc</artifactId>
        <version>5.0.8</version>
    </dependency>
    <!-- JDBC Driver for your database -->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <version>8.0.23</version>
    </dependency>
</dependencies>
```

```java
import javax.transaction.UserTransaction;
import com.atomikos.icatch.jta.UserTransactionManager;

public class TwoPhaseCommitExample {

    public static void main(String[] args) {
        UserTransactionManager transactionManager = new UserTransactionManager();

        try {
            transactionManager.init(); // 트랜잭션 매니저 초기화
            UserTransaction tx = transactionManager.getTransaction();

            tx.begin(); // 트랜잭션 시작

            // 첫 번째 데이터베이스 리소스에 대한 작업
            // 예: dataSource1.getConnection().prepareStatement("SQL 쿼리").executeUpdate();

            // 두 번째 데이터베이스 리소스에 대한 작업
            // 예: dataSource2.getConnection().prepareStatement("SQL 쿼리").executeUpdate();

            tx.commit(); // 모든 작업이 성공적으로 완료되면 커밋
        } catch (Exception e) {
            try {
                tx.rollback(); // 오류 발생 시 롤백
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            e.printStackTrace();
        } finally {
            try {
                transactionManager.close(); // 트랜잭션 매니저 종료
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

이 코드는 두 개의 데이터베이스 리소스에 대한 작업을 포함하는 트랜잭션을 시작하고, 모든 작업이 성공적으로 완료되면 커밋하거나, 오류가 발생하면 롤백하는 기본적인 2PC 흐름을 보여줍니다. 실제 환경에서는 데이터 소스 구성, 오류 처리, 리소스 정리 등 추가적인 고려사항이 있을 수 있습니다.

Atomikos와 같은 트랜잭션 매니저를 사용하면, 개발자는 2PC 프로토콜의 복잡한 세부사항을 직접 처리할 필요 없이, 분산 트랜잭션을 효율적으로 관리할 수 있습니다. 그러나 이러한 종류의 처리는 성능 오버헤드를 동반할 수 있으므로, 필요한 경우에만 사용해야 합니다.

# TCC (Try-Confirm/Cancel)

* [REST 기반의 간단한 분산 트랜잭션 구현 – 1편 TCC 개관](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-1%ED%8E%B8/)
  * [src](https://github.com/YooYoungmo/article-tcc)
  * [REST 기반의 간단한 분산 트랜잭션 구현 - 2편 TCC Cancel, Timeout](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-2%ED%8E%B8-tcc-cancel-timeout/)
  * [REST 기반의 간단한 분산 트랜잭션 구현 - 3편 TCC Confirm(Eventual Consistency)](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-3%ED%8E%B8-tcc-confirmeventual-consistency/)
  * [REST 기반의 간단한 분산 트랜잭션 구현 - 4편 REST Retry](https://www.popit.kr/rest-%EA%B8%B0%EB%B0%98%EC%9D%98-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B6%84%EC%82%B0-%ED%8A%B8%EB%9E%9C%EC%9E%AD%EC%85%98-%EA%B5%AC%ED%98%84-4%ED%8E%B8-rest-retry/)

## Sequences

![](tcc.png)

1. **order** : User request order. `Order Service` is Transaction Coordinator.

2. **try reduce stock** : `POST /api/v1/stocks HTTP/1.1`
   
  * reponse body 
    ```json
    {
      url: "http://localhost:8081/api/v1/stocks/1",
      expires: "2020-08-22T09:00:00.000"
    }
    ```
  * Create a record with `status = reserved` in `reserved_stock` table. 
  
    | id | created | resources | status |
    |--|--|--|--|
    | 1 | 2020-08-22 09:00:00.000 | {"productid": "0001", "adjustmentType": "REDUCE", "qty": 10} | "reserved" |

3. **try withdraw payment** : `POST /api/v1/payments HTTP/1.1`
  * reponse body 
    ```json
    {
      url: "http://localhost:8081/api/v1/payments/1",
      expires: "2020-08-22T09:00:00.000"
    }
    ``` 
 * Create a record with `status = reserved` in `reserved_payment` table. 

4. create order 
   * Create a record in `order` table.

5. **confirm reduce stock** : `PUT /api/v1/stocks/{id} HTTP/1.1`
  1. publish reduce stock msg
  2. consume reduce stock msg
  3. **reduce stock** : Update a record in `reserved_stock, stock` table.

     | id | created | resources | status |
     |--|--|--|--|
     | 1 | 2020-08-22 09:00:00.000 | {"productid": "0001", "adjustmentType": "REDUCE", "qty": 10} | "confirmed" |

     | id | prodct_id | qty |
     |--|--|--|
     | 1 | "0001" | 0 |
     | 2 | "0002" | 20 |
     | 3 | "0003" | 30 |

6. **confirm withdraw payment** : `PUT /api/v1/payments/{id} HTTP/1.1`
  4. publish withdraw payment
  5. consume withdraw payment
  6. **withdraw payment** : Update a record in `payment` table.

## Exceptions

* When it fails on `2, 3` stage.
  * Retry is a good solution.
* When it fails on `4` stage
  * `Order Service` send Cancel request `DELETE /api/v1/sotkcs/{id} HTTP/1.1` to `Stock Service` and `DELETE /api/v1/payments/{id} HTTP/1.1` to `Payment Service`.
* When it fails on `4` stage and Cancel request also fails.
  * `Stock Service, Payment Service` can handle with timeout. Especially `Stock Service, Payment Service` should have a `expires` column in their reserved tables.
  
    | id | expires | created | resources | status |
    |--|--|--|--|--|
    | 1 | 2020-08-22 09:00:03.000 | 2020-08-22 09:00:00.000 | {"productid": "0001", "adjustmentType": "REDUCE", "qty": 10} | "confirmed" |
* `Stock Service, Payment Service` should delete records of reserved tables periodically.
* When it fails on `5, 6` stage.
  * Kafka can handle this. `Stock Service, Payment Service` publish, consume messages and try to do `reduce sotck, withdraw payment` until they succeed repeatedly. This means Eventual Consistency.

## Summary

**Order Service** is a `Transaction Coordinator`. [Kafka](/kafka/README.md) can handle Eventual Consistency.

## JAVA Example

Try-Confirm/Cancel (TCC) 패턴은 분산 시스템에서 일관성 있는 트랜잭션 처리를 위한 패턴 중 하나입니다. 이 패턴은 두 단계 커밋(2PC) 프로토콜과 유사하나, 각 단계가 명시적으로 "시도(Try)", "확인(Confirm)", 그리고 "취소(Cancel)"로 구분됩니다. TCC는 주로 롱 트랜잭션에서 사용되며, 각 단계는 다음과 같은 역할을 합니다:

- 시도(Try): 자원 예약 및 검증 단계로, 실제 작업을 수행하기 전에 필요한 자원을 예약하고, 트랜잭션이 성공할 수 있는지 확인합니다.
- 확인(Confirm): 모든 Try 단계가 성공적으로 완료되면, 실제 작업을 커밋합니다.
- 취소(Cancel): 어떤 Try 단계에서라도 실패하면, 이미 예약된 모든 자원을 해제하고 작업을 롤백합니다.

Java로 TCC 패턴을 구현하는 기본적인 예제를 아래에 제시합니다. 이 예제는 두 개의 서비스(ServiceA와 ServiceB)를 사용하여 간단한 TCC 트랜잭션을 구현합니다. 실제 애플리케이션에서는 이러한 서비스가 다양한 마이크로서비스나 시스템으로 구현될 수 있습니다.

```java
// Service.java
// 이 인터페이스는 Try, Confirm, Cancel 메소드를 포함합니다.
public interface Service {
    boolean tryAction();
    void confirmAction();
    void cancelAction();
}

// ServiceA와 ServiceB에 대한 구현을 제공합니다. 각 메소드는 단순화를 위해 boolean 값을 반환하며, 
// 실제 환경에서는 더 복잡한 로직을 포함할 수 있습니다.
// ServiceA.java
public class ServiceA implements Service {
    public boolean tryAction() {
        System.out.println("ServiceA tryAction executed");
        // 자원 예약 및 검증 로직
        return true; // 성공적으로 수행된 경우
    }

    public void confirmAction() {
        System.out.println("ServiceA confirmAction executed");
        // 실제 작업 커밋
    }

    public void cancelAction() {
        System.out.println("ServiceA cancelAction executed");
        // 예약된 자원 해제 및 롤백
    }
}

// ServiceB.java
public class ServiceB implements Service {
    public boolean tryAction() {
        System.out.println("ServiceB tryAction executed");
        // 자원 예약 및 검증 로직
        return true; // 성공적으로 수행된 경우
    }

    public void confirmAction() {
        System.out.println("ServiceB confirmAction executed");
        // 실제 작업 커밋
    }

    public void cancelAction() {
        System.out.println("ServiceB cancelAction executed");
        // 예약된 자원 해제 및 롤백
    }
}

// TccTransactionManager.java
// 마지막으로, TCC 트랜잭션을 관리하는 간단한 관리자 클래스를 구현합니다. 
// 이 클래스는 모든 서비스의 Try 단계를 실행하고, 성공하면 Confirm 단계를, 실패하면 Cancel 단계를 실행합니다.
import java.util.ArrayList;
import java.util.List;

public class TccTransactionManager {
    private List<Service> services = new ArrayList<>();

    public void addService(Service service) {
        services.add(service);
    }

    public boolean executeTransaction() {
        // Try 단계 실행
        for (Service service : services) {
            if (!service.tryAction()) {
                // 실패 시 Cancel 단계 실행
                cancelTransaction();
                return false;
            }
        }

        // 모든 Try 단계 성공 시 Confirm 단계 실행
        confirmTransaction();
        return true;
    }

    private void confirmTransaction() {
        for (Service service : services) {
            service.confirmAction();
        }
    }

    private void cancelTransaction() {
        for (Service service : services) {
            service.cancelAction();
        }
    }

    public static void main(String[] args) {
        TccTransactionManager manager = new TccTransactionManager();
        manager.addService(new ServiceA());
        manager.addService(new ServiceB());

        if (manager.executeTransaction()) {
            System.out.println("Transaction succeeded");
        } else {
            System.out.println("Transaction failed");
        }
    }
}
```

이 예제는 TCC 패턴의 기본 개념과 단계를 보여줍니다. 실제 분산 시스템에서는 통신 실패, 타임아웃 처리, 상태 관리 등을 고려해야 하며, 이러한 복잡성을 관리하기 위해 Saga 패턴과 같은 다른 패턴을 사용할 수도 있습니다.

# 2PC vs TC/C

| Phase | 2PC | TC/C |
|--|--|--|
| First Phase | Local transactions are not done yet | All local transactions completed, committed or canceled |
| Second Phase: **success** | Commit all local transactions | Execute new local transactions if needed |
| Third Phase: **fail** | Cancel all local transactions | Reverse the side effect of the already committed transaction, or called "undo" |

# SAGA

* [Orchestration vs. Choreography](https://stackoverflow.com/questions/4127241/orchestration-vs-choreography)
* [7. Introduction to Saga | baeldung](https://www.baeldung.com/cs/saga-pattern-microservices#introduction-to-saga)
  * [A Guide to Transactions Across Microservices | baeldung](https://www.baeldung.com/transactions-across-microservices)

----

## SAGA Overview

* [Saga Pattern in Microservices | baeldung](https://www.baeldung.com/cs/saga-pattern-microservices)
* [Distributed Transactions in Microservices with Kafka Streams and Spring Boot](https://piotrminkowski.com/2022/01/24/distributed-transactions-in-microservices-with-kafka-streams-and-spring-boot/)
  * [src](https://github.com/piomin/sample-spring-kafka-microservices)

SAGA 는 global transaction 을 local transaction 으로 나누고 순서대로 처리하는 방법이다. `TC/C` 는 local transaction 이 순서대로 처리되지 않는다. 예외 사항이 더 많아서 구현이 어렵다. 

![](img/saga_flow.png)

global transaction 은 `Create Order, Process Payment, Update Inventory, Deliver Order` 와 같은 local transaction 들로 나누어져 있다. 

만약 `Process Payment` transaction 이 실패한다면 `Reverse Payment, Cancel Order` 순서로 Compensating Transaction 을 실행한다.

Compensating Transaction 이 실패한다면 어딘가에 저장해 놓고 Eventual Consistent 하게 처리한다. 예를 들어 Kafka Topic 에 실패한 Compensating Transaction 을 저장해 놓고 성공할 때까지 재시도 한다. 따라서 재시도 해야할 task 는 idempotent, retryable 해야 한다.

SAGA 는 **Choreography SAGA, Orchestration SAGA** 와 같이 2 종류가 있다. **Choreography SAGA** 는 transaction 성공여부 판단을 각 service 에서 나누어 한다. **Orchestration SAGA** 는 transaction 성공여부 판단을 한 곳에서 한다.

## Choreography SAGA

* [Choreography-based sagas example @ github](https://github.com/eventuate-tram/eventuate-tram-examples-customers-and-orders)
  
----  

다음은 Baeldung 의 Choreography SAGA Architecture 이다.

![](https://www.baeldung.com/wp-content/uploads/sites/4/2021/04/saga-coreography.png)

Saga Execution Coordinator 는 Framewok 와 같다. 각 microservice 에 embed 되었다고 생각하자. micro service 들은 SEC (SAGA Execution Component) 와 message 들을 주고 받는다. 그리고 transaction 혹은 compensation transaction 을 수행한다.

다음은 Tx2 가 실패했을 때의 처리과정이다.

![](https://www.baeldung.com/wp-content/uploads/sites/4/2021/04/saga-coreography-2.png)

다음은 Chris Richardson 의 Choregography SAGA Architecture 이다.

![](choreography_saga.jpg)

![](https://chrisrichardson.net/i/sagas/Create_Order_Saga.png)

### Java Example

Choreography SAGA 패턴은 분산 시스템에서 각 마이크로서비스가 독립적으로 자신의 로컬 트랜잭션을 관리하고, 필요한 보상 트랜잭션(롤백 로직)을 실행하여 전체 비즈니스 트랜잭션의 일관성을 유지하는 방식입니다. 이 패턴에서는 중앙 집중식 조정자(coordinator) 없이 각 서비스가 다음 서비스의 트랜잭션을 트리거합니다. 이 예제에서는 간단한 주문 처리 시스템을 구현하여 Choreography SAGA 패턴을 보여드립니다.

이 시스템은 세 개의 마이크로서비스로 구성됩니다:

- Order Service: 주문을 생성합니다.
- Payment Service: 결제를 처리합니다.
- Inventory Service: 재고를 감소시킵니다.

각 서비스는 성공 시 다음 서비스를 호출하고, 실패 시 보상 트랜잭션을 실행합니다. 이 예제는 각 서비스의 핵심 로직과 보상 로직을 단순화하여 설명합니다.

```java
// OrderService.java
public class OrderService {
    public void createOrder() {
        try {
            System.out.println("Creating order");
            // 여기서 주문 생성 로직을 구현합니다.
            PaymentService paymentService = new PaymentService();
            paymentService.processPayment();
        } catch (Exception e) {
            System.out.println("Order creation failed");
            // 필요한 보상 로직을 구현합니다.
        }
    }
}

// PaymentService.java
public class PaymentService {
    public void processPayment() {
        try {
            System.out.println("Processing payment");
            // 여기서 결제 처리 로직을 구현합니다.
            InventoryService inventoryService = new InventoryService();
            inventoryService.updateInventory();
        } catch (Exception e) {
            System.out.println("Payment processing failed");
            // 보상 로직을 실행합니다. 예를 들어, 결제를 취소합니다.
        }
    }
}

// InventoryService.java
public class InventoryService {
    public void updateInventory() {
        try {
            System.out.println("Updating inventory");
            // 여기서 재고 감소 로직을 구현합니다.
        } catch (Exception e) {
            System.out.println("Inventory update failed");
            // 보상 로직을 실행합니다. 예를 들어, 재고 수량을 복구합니다.
        }
    }
}

// SagaChoreographyExample
public class SagaChoreographyExample {
    public static void main(String[] args) {
        OrderService orderService = new OrderService();
        orderService.createOrder();
    }
}
```

이 예제에서는 각 서비스가 독립적으로 실행되며, 각 단계가 성공적으로 완료되면 다음 서비스를 호출합니다. 만약 어느 단계에서 실패하면, 해당 서비스는 자체적으로 정의된 보상 로직을 실행하여 시스템을 일관된 상태로 복구합니다. 실제 시나리오에서는 이벤트 버스(예: Apache Kafka, RabbitMQ)를 사용하여 서비스 간 비동기 통신을 구현하고, 각 서비스는 이벤트를 구독하여 다음 작업을 트리거하거나 보상 로직을 실행할 수 있습니다. Choreography SAGA 패턴은 각 서비스가 자율적으로 행동하며 복잡한 중앙 집중식 조정 로직 없이도 분산 트랜잭션의 일관성을 유지할 수 있게 해줍니다.

## Orchestration SAGA

* [orchestration-based-sagas example @ github](https://github.com/eventuate-tram/eventuate-tram-sagas-examples-customers-and-orders)
* [Distributed Transactions in Microservices with Kafka Streams and Spring Boot](https://piotrminkowski.com/2022/01/24/distributed-transactions-in-microservices-with-kafka-streams-and-spring-boot/)
  * Orchestration SAGA Implementation with Kafka Streams and Spring Boot
  * [src](https://github.com/piomin/sample-spring-kafka-microservices)
  
----

다음은 Baeldung 의 Orchestration SAGA Architecture 이다.

![](https://www.baeldung.com/wp-content/uploads/sites/4/2021/04/saga-orchestration.png)

SEC (SAGA Execution Component) 가 직접 Compensation Transaction 을 수행한다???

Chris Richardson 의 Orchestration SAGA Architecture.

![](orchestration_saga.jpg)

![](https://chrisrichardson.net/i/sagas/Create_Order_Saga_Orchestration.png)

### Java Example

Orchestration SAGA 패턴은 복잡한 분산 시스템에서 전체 트랜잭션을 관리하는 중앙 집중식 서비스나 컴포넌트(orchestrator)를 사용하여 각 마이크로서비스의 로컬 트랜잭션을 순차적 또는 병렬로 실행하는 방식입니다. Orchestrator는 전체 비즈니스 트랜잭션의 진행 상태를 관리하고, 필요한 경우 보상 트랜잭션(롤백 로직)을 실행하여 일관성을 유지합니다.

이 예제에서는 주문 처리 시스템을 Orchestrator와 함께 구현하여 Orchestration SAGA 패턴을 보여드립니다. 시스템은 세 가지 주요 서비스로 구성됩니다:

- Order Service: 주문을 관리합니다.
- Payment Service: 결제를 처리합니다.
- Inventory Service: 재고를 관리합니다.

Orchestrator는 이 서비스들을 조정합니다.

```java
// OrderService.java
public interface OrderService {
    boolean createOrder();
    void cancelOrder();
}
// PaymentService.java
public interface PaymentService {
    boolean processPayment();
    void refundPayment();
}
// InventoryService.java
public interface InventoryService {
    boolean updateInventory();
    void revertInventory();
}

// SagaOrchestrator.java
// Orchestrator는 서비스를 조정하고, 각 서비스의 성공/실패 여부에 따라 
// 다음 단계를 실행하거나 보상 트랜잭션을 실행합니다.
public class SagaOrchestrator {
    private OrderService orderService;
    private PaymentService paymentService;
    private InventoryService inventoryService;

    public SagaOrchestrator(OrderService orderService, PaymentService paymentService, InventoryService inventoryService) {
        this.orderService = orderService;
        this.paymentService = paymentService;
        this.inventoryService = inventoryService;
    }

    public void executeSaga() {
        if (orderService.createOrder()) {
            if (paymentService.processPayment()) {
                if (!inventoryService.updateInventory()) {
                    System.out.println("Inventory update failed, initiating compensation...");
                    paymentService.refundPayment();
                    orderService.cancelOrder();
                }
            } else {
                System.out.println("Payment processing failed, initiating compensation...");
                orderService.cancelOrder();
            }
        }
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        OrderService orderService = new OrderServiceImpl(); // 가상의 구현체
        PaymentService paymentService = new PaymentServiceImpl(); // 가상의 구현체
        InventoryService inventoryService = new InventoryServiceImpl(); // 가상의 구현체

        SagaOrchestrator orchestrator = new SagaOrchestrator(orderService, paymentService, inventoryService);
        orchestrator.executeSaga();
    }
}
```

Orchestrator는 각 서비스의 실행을 조정하고, 각 단계에서 발생할 수 있는 실패를 처리하기 위해 보상 트랜잭션을 실행합니다. 이 패턴은 복잡한 비즈니스 트랜잭션과 롤백 시나리오를 처리할 수 있으며, 각 서비스의 독립성을 유지하면서 전체 트랜잭션의 일관성을 보장합니다. Orchestrator의 구현은 분산 시스템의 복잡성을 관리하는 중요한 역할을 하며, 비즈니스 로직과 롤백 메커니즘을 명확하게 정의해야 합니다.

# TC/C vs Sagas

| Item | TC/C | SAGAS |
|---|---|---|
| Compensating action | In Cancel Phase | In rollback phase |
| Central coordination | Yes | Yes (Orchestration SAGA) |
| Operation execution order | any | linear |
| Parallel execution possibility | Yes | No (linear) |
| Could see the partial inconsistent status | Yes | Yes |
| Application or database logic | Application | Application |

# Conclusion

SAGA 가 제일 그럴듯하다. 

Choreography SAGA 는 greenfield project 에 적당하다. 처음부터 project 를 시작한다면 할 만하다는 의미이다. 많은 micro service 제작자들과 local transaction 의 상태등을 포함해서 협의를 해야하기 때문이다.

Orchestration SAGA 는 brownfield project 에 적당하다. 이미 진행된 project 에 적용할 만하다. 많은 micro service 제작자들과 협의해야할 내용이 Choreography SAGA 에 비해 적다. local transaction 의 상태는 orchestrator 만 알아도 된다. 
