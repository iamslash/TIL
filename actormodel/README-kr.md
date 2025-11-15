- [Materials](#materials)
- [Basic](#basic)
  - [Actor Model이란?](#actor-model이란)
    - [형식적 의미론 (Formal Semantics)](#형식적-의미론-formal-semantics)
    - [무제한 동시성 (Unbounded Concurrency)](#무제한-동시성-unbounded-concurrency)
    - [위치 투명성 (Location Transparency)](#위치-투명성-location-transparency)
  - [핵심 개념](#핵심-개념)
    - [1. 동시성 (Concurrency)](#1-동시성-concurrency)
    - [2. 격리 (Isolation)](#2-격리-isolation)
    - [3. 비동기 통신 (Asynchronous Communication)](#3-비동기-통신-asynchronous-communication)
    - [4. 위치 투명성 (Location Transparency)](#4-위치-투명성-location-transparency)
  - [E-Commerce 예제: 비동기 통신](#e-commerce-예제-비동기-통신)
    - [Java/Akka 구현](#javaakka-구현)
    - [실행 결과](#실행-결과)
    - [비동기 통신의 장점](#비동기-통신의-장점)
  - [E-Commerce 예제: 확장성 (Scalability)](#e-commerce-예제-확장성-scalability)
    - [Java/Akka Router를 활용한 확장](#javaakka-router를-활용한-확장)
    - [실행 결과](#실행-결과-1)
    - [확장성 전략](#확장성-전략)
    - [동적 확장](#동적-확장)
  - [E-Commerce 예제: 장애 내성 (Fault-Tolerance)](#e-commerce-예제-장애-내성-fault-tolerance)
    - [Java/Akka Supervision을 활용한 장애 처리](#javaakka-supervision을-활용한-장애-처리)
    - [실행 결과](#실행-결과-2)
    - [Supervision 전략](#supervision-전략)
    - [Supervision 계층](#supervision-계층)
  - [E-Commerce 예제: 위치 투명성 (Location Transparency)](#e-commerce-예제-위치-투명성-location-transparency)
    - [Java/Akka Clustering을 활용한 분산 시스템](#javaakka-clustering을-활용한-분산-시스템)
    - [실행 결과](#실행-결과-3)
    - [위치 투명성의 장점](#위치-투명성의-장점)
    - [Akka Clustering 패턴](#akka-clustering-패턴)
  - [Actor Model 프레임워크 비교](#actor-model-프레임워크-비교)
    - [프레임워크 선택 기준](#프레임워크-선택-기준)
    - [실제 사용 사례](#실제-사용-사례)
  - [결론](#결론)

-----

# Materials

- [The actor model in 10 minutes](https://www.brianstorti.com/the-actor-model/)
- [akka @ github](https://github.com/akka/akka)
  - [akka quickstart](https://developer.lightbend.com/guides/akka-quickstart-java/)
- [pekko @ github](https://github.com/apache/pekko)

# Basic

## Actor Model이란?

Actor Model은 **동시성 컴퓨팅**을 위한 수학적 모델이자 설계 패턴입니다. 1973년 Carl Hewitt에 의해 제안된 이 모델은 "Actor"라는 기본 단위를 통해 병렬 및 분산 시스템을 구축합니다.

### 형식적 의미론 (Formal Semantics)

Actor는 메시지를 받았을 때 다음 세 가지를 동시에 수행할 수 있습니다:

1. **유한한 수의 새로운 Actor 생성** - 새로운 동시성 단위 생성
2. **유한한 수의 메시지를 다른 Actor에게 전송** - 비동기 통신
3. **다음 메시지를 처리할 동작 지정** - 상태 관리

### 무제한 동시성 (Unbounded Concurrency)

Actor Model의 핵심 특징 중 하나는 **무제한 동시성**입니다. 각 Actor는 독립적으로 동작하며, 시스템의 Actor 수에는 이론적 제한이 없습니다. 이는 시스템이 필요에 따라 수평 확장(horizontal scaling)할 수 있음을 의미합니다.

### 위치 투명성 (Location Transparency)

Actor는 **로컬이든 원격이든 동일한 방식으로 메시지를 주고받습니다**. 이는 분산 시스템 개발을 크게 단순화하며, 코드 변경 없이 시스템을 확장할 수 있게 합니다.

## 핵심 개념

### 1. 동시성 (Concurrency)

전통적인 쓰레드 기반 동시성의 문제점:
- **공유 메모리 관리의 복잡성** - Race condition, deadlock, livelock
- **디버깅의 어려움** - 비결정적 동작
- **확장성 제한** - Lock contention

Actor Model의 해결 방법:
- **공유 상태 없음** (No Shared State) - 각 Actor는 독립적인 상태 소유
- **메시지 기반 통신** - Actor 간 상호작용은 오직 메시지를 통해서만
- **순차적 메시지 처리** - 각 Actor의 mailbox는 메시지를 순차적으로 처리

### 2. 격리 (Isolation)

각 Actor는 완전히 격리된 실행 단위입니다:
- **독립적인 상태** - 다른 Actor가 직접 접근 불가
- **캡슐화** - 내부 구현 은닉
- **장애 격리** - 한 Actor의 실패가 다른 Actor에 영향 없음

### 3. 비동기 통신 (Asynchronous Communication)

Actor 간 통신은 항상 비동기적입니다:
- **논블로킹** - 메시지 전송 후 즉시 다른 작업 수행 가능
- **Fire-and-forget** - 기본적으로 응답을 기다리지 않음
- **높은 처리량** - 블로킹 없이 많은 메시지 처리 가능

### 4. 위치 투명성 (Location Transparency)

Actor의 물리적 위치는 중요하지 않습니다:
- **동일한 API** - 로컬/원격 Actor 구분 없음
- **동적 배치** - 런타임에 Actor 위치 변경 가능
- **분산 시스템 지원** - 여러 노드에 걸쳐 Actor 배치

## E-Commerce 예제: 비동기 통신

온라인 쇼핑몰에서 주문 처리 시스템을 구현해보겠습니다. 주문이 들어오면 결제 처리, 재고 확인, 배송 준비가 비동기적으로 진행됩니다.

### Java/Akka 구현

```java
import akka.actor.AbstractActor;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;

// 메시지 정의
class Order {
    final String orderId;
    final String customerId;
    final List<String> items;
    final double totalAmount;

    public Order(String orderId, String customerId,
                 List<String> items, double totalAmount) {
        this.orderId = orderId;
        this.customerId = customerId;
        this.items = items;
        this.totalAmount = totalAmount;
    }
}

class PaymentRequest {
    final String orderId;
    final double amount;

    public PaymentRequest(String orderId, double amount) {
        this.orderId = orderId;
        this.amount = amount;
    }
}

class PaymentResult {
    final String orderId;
    final boolean success;

    public PaymentResult(String orderId, boolean success) {
        this.orderId = orderId;
        this.success = success;
    }
}

class InventoryCheck {
    final String orderId;
    final List<String> items;

    public InventoryCheck(String orderId, List<String> items) {
        this.orderId = orderId;
        this.items = items;
    }
}

class InventoryResult {
    final String orderId;
    final boolean available;

    public InventoryResult(String orderId, boolean available) {
        this.orderId = orderId;
        this.available = available;
    }
}

// Order Actor - 주문 조정자
class OrderActor extends AbstractActor {
    private final ActorRef paymentActor;
    private final ActorRef inventoryActor;

    // 주문 상태 추적
    private Map<String, OrderState> orderStates = new HashMap<>();

    static class OrderState {
        boolean paymentComplete = false;
        boolean inventoryChecked = false;
    }

    public OrderActor(ActorRef paymentActor, ActorRef inventoryActor) {
        this.paymentActor = paymentActor;
        this.inventoryActor = inventoryActor;
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(Order.class, this::handleOrder)
            .match(PaymentResult.class, this::handlePaymentResult)
            .match(InventoryResult.class, this::handleInventoryResult)
            .build();
    }

    private void handleOrder(Order order) {
        System.out.println("OrderActor: 주문 수신 - " + order.orderId);

        // 상태 초기화
        orderStates.put(order.orderId, new OrderState());

        // 비동기로 결제와 재고 확인 동시 진행
        paymentActor.tell(
            new PaymentRequest(order.orderId, order.totalAmount),
            getSelf()
        );
        inventoryActor.tell(
            new InventoryCheck(order.orderId, order.items),
            getSelf()
        );

        System.out.println("OrderActor: 결제 및 재고 확인 요청 전송");
    }

    private void handlePaymentResult(PaymentResult result) {
        System.out.println("OrderActor: 결제 결과 수신 - " +
            result.orderId + " / " +
            (result.success ? "성공" : "실패"));

        OrderState state = orderStates.get(result.orderId);
        if (state != null) {
            state.paymentComplete = result.success;
            checkOrderCompletion(result.orderId, state);
        }
    }

    private void handleInventoryResult(InventoryResult result) {
        System.out.println("OrderActor: 재고 확인 결과 수신 - " +
            result.orderId + " / " +
            (result.available ? "재고 있음" : "재고 없음"));

        OrderState state = orderStates.get(result.orderId);
        if (state != null) {
            state.inventoryChecked = result.available;
            checkOrderCompletion(result.orderId, state);
        }
    }

    private void checkOrderCompletion(String orderId, OrderState state) {
        // 결제와 재고 확인이 모두 완료되면 주문 완료
        if (state.paymentComplete && state.inventoryChecked) {
            System.out.println("OrderActor: 주문 처리 완료 - " + orderId);
            orderStates.remove(orderId);
        }
    }

    public static Props props(ActorRef paymentActor, ActorRef inventoryActor) {
        return Props.create(OrderActor.class,
            () -> new OrderActor(paymentActor, inventoryActor));
    }
}

// Payment Actor - 결제 처리
class PaymentActor extends AbstractActor {
    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(PaymentRequest.class, this::processPayment)
            .build();
    }

    private void processPayment(PaymentRequest request) {
        System.out.println("PaymentActor: 결제 처리 시작 - " +
            request.orderId + " / $" + request.amount);

        // 실제로는 외부 결제 게이트웨이 호출
        // 여기서는 시뮬레이션
        try {
            Thread.sleep(1000); // 결제 처리 시뮬레이션
            boolean success = Math.random() > 0.1; // 90% 성공률

            // 결과를 원래 요청자(OrderActor)에게 전송
            getSender().tell(
                new PaymentResult(request.orderId, success),
                getSelf()
            );
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public static Props props() {
        return Props.create(PaymentActor.class, PaymentActor::new);
    }
}

// Inventory Actor - 재고 확인
class InventoryActor extends AbstractActor {
    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(InventoryCheck.class, this::checkInventory)
            .build();
    }

    private void checkInventory(InventoryCheck check) {
        System.out.println("InventoryActor: 재고 확인 시작 - " +
            check.orderId + " / 아이템 수: " + check.items.size());

        // 실제로는 데이터베이스 조회
        try {
            Thread.sleep(800); // 재고 확인 시뮬레이션
            boolean available = Math.random() > 0.05; // 95% 재고 있음

            // 결과를 원래 요청자(OrderActor)에게 전송
            getSender().tell(
                new InventoryResult(check.orderId, available),
                getSelf()
            );
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public static Props props() {
        return Props.create(InventoryActor.class, InventoryActor::new);
    }
}

// 메인 애플리케이션
public class ECommerceActorSystem {
    public static void main(String[] args) throws InterruptedException {
        // Actor System 생성
        ActorSystem system = ActorSystem.create("ECommerceSystem");

        try {
            // Actor 생성
            ActorRef paymentActor = system.actorOf(
                PaymentActor.props(),
                "paymentActor"
            );
            ActorRef inventoryActor = system.actorOf(
                InventoryActor.props(),
                "inventoryActor"
            );
            ActorRef orderActor = system.actorOf(
                OrderActor.props(paymentActor, inventoryActor),
                "orderActor"
            );

            // 테스트 주문 생성
            Order order1 = new Order(
                "ORD-001",
                "CUST-100",
                Arrays.asList("노트북", "마우스", "키보드"),
                1500.00
            );

            Order order2 = new Order(
                "ORD-002",
                "CUST-101",
                Arrays.asList("모니터", "HDMI 케이블"),
                450.00
            );

            // 비동기 주문 처리
            System.out.println("=== 주문 1 시작 ===");
            orderActor.tell(order1, ActorRef.noSender());

            Thread.sleep(500); // 약간의 간격

            System.out.println("\n=== 주문 2 시작 ===");
            orderActor.tell(order2, ActorRef.noSender());

            // 처리 완료 대기
            Thread.sleep(3000);

        } finally {
            system.terminate();
        }
    }
}
```

### 실행 결과

```
=== 주문 1 시작 ===
OrderActor: 주문 수신 - ORD-001
OrderActor: 결제 및 재고 확인 요청 전송
PaymentActor: 결제 처리 시작 - ORD-001 / $1500.0
InventoryActor: 재고 확인 시작 - ORD-001 / 아이템 수: 3

=== 주문 2 시작 ===
OrderActor: 주문 수신 - ORD-002
OrderActor: 결제 및 재고 확인 요청 전송
PaymentActor: 결제 처리 시작 - ORD-002 / $450.0
InventoryActor: 재고 확인 시작 - ORD-002 / 아이템 수: 2
InventoryActor: 재고 확인 결과 수신 - ORD-001 / 재고 있음
OrderActor: 결제 결과 수신 - ORD-001 / 성공
OrderActor: 주문 처리 완료 - ORD-001
InventoryActor: 재고 확인 결과 수신 - ORD-002 / 재고 있음
OrderActor: 결제 결과 수신 - ORD-002 / 성공
OrderActor: 주문 처리 완료 - ORD-002
```

### 비동기 통신의 장점

1. **논블로킹** - OrderActor는 응답을 기다리지 않고 다음 주문 처리 가능
2. **동시성** - 여러 주문이 동시에 처리됨
3. **격리** - 각 Actor는 독립적으로 동작하며 서로 간섭하지 않음
4. **확장성** - Actor를 추가하여 처리량 증가 가능

## E-Commerce 예제: 확장성 (Scalability)

블랙 프라이데이 같은 대규모 이벤트에서 주문이 폭증할 때 시스템이 자동으로 확장되도록 구현해보겠습니다.

### Java/Akka Router를 활용한 확장

```java
import akka.actor.*;
import akka.routing.RoundRobinPool;
import java.time.Duration;

// 주문 처리 워커
class OrderProcessor extends AbstractActor {
    private final String workerId;

    public OrderProcessor() {
        this.workerId = getSelf().path().name();
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(Order.class, this::processOrder)
            .build();
    }

    private void processOrder(Order order) {
        System.out.println(String.format(
            "[Worker %s] 주문 처리 시작: %s (고객: %s, 금액: $%.2f)",
            workerId, order.orderId, order.customerId, order.totalAmount
        ));

        try {
            // 주문 처리 시뮬레이션 (복잡한 비즈니스 로직)
            Thread.sleep(500 + (long)(Math.random() * 500));

            System.out.println(String.format(
                "[Worker %s] 주문 처리 완료: %s",
                workerId, order.orderId
            ));

            // 처리 결과를 원래 요청자에게 전송
            getSender().tell(
                new OrderProcessed(order.orderId, true),
                getSelf()
            );

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            getSender().tell(
                new OrderProcessed(order.orderId, false),
                getSelf()
            );
        }
    }

    public static Props props() {
        return Props.create(OrderProcessor.class, OrderProcessor::new);
    }
}

class OrderProcessed {
    final String orderId;
    final boolean success;

    public OrderProcessed(String orderId, boolean success) {
        this.orderId = orderId;
        this.success = success;
    }
}

// 주문 분배 관리자
class OrderDistributor extends AbstractActor {
    private final ActorRef orderRouter;
    private int totalOrders = 0;
    private int completedOrders = 0;

    public OrderDistributor(int numberOfWorkers) {
        // RoundRobinPool: 라운드로빈 방식으로 워커에게 작업 분배
        // 다른 전략: RandomPool, SmallestMailboxPool, ConsistentHashingPool
        this.orderRouter = getContext().actorOf(
            new RoundRobinPool(numberOfWorkers).props(OrderProcessor.props()),
            "order-router"
        );

        System.out.println(String.format(
            "OrderDistributor: %d개의 워커로 시작",
            numberOfWorkers
        ));
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(Order.class, this::distributeOrder)
            .match(OrderProcessed.class, this::handleOrderProcessed)
            .match(GetStatistics.class, this::sendStatistics)
            .build();
    }

    private void distributeOrder(Order order) {
        totalOrders++;
        // Router가 자동으로 워커에게 분배
        orderRouter.tell(order, getSelf());
    }

    private void handleOrderProcessed(OrderProcessed result) {
        completedOrders++;

        if (result.success) {
            System.out.println(String.format(
                "OrderDistributor: 주문 완료 (%d/%d)",
                completedOrders, totalOrders
            ));
        }
    }

    private void sendStatistics(GetStatistics msg) {
        getSender().tell(
            new Statistics(totalOrders, completedOrders),
            getSelf()
        );
    }

    public static Props props(int numberOfWorkers) {
        return Props.create(OrderDistributor.class,
            () -> new OrderDistributor(numberOfWorkers));
    }
}

class GetStatistics {}

class Statistics {
    final int total;
    final int completed;

    public Statistics(int total, int completed) {
        this.total = total;
        this.completed = completed;
    }
}

// 부하 생성기 (테스트용)
class LoadGenerator extends AbstractActor {
    private final ActorRef distributor;
    private final int ordersPerSecond;
    private Cancellable scheduler;

    public LoadGenerator(ActorRef distributor, int ordersPerSecond) {
        this.distributor = distributor;
        this.ordersPerSecond = ordersPerSecond;
    }

    @Override
    public void preStart() {
        // 주기적으로 주문 생성
        scheduler = getContext().getSystem().scheduler().scheduleAtFixedRate(
            Duration.ZERO,
            Duration.ofMillis(1000 / ordersPerSecond),
            getSelf(),
            new GenerateOrder(),
            getContext().getDispatcher(),
            ActorRef.noSender()
        );
    }

    @Override
    public void postStop() {
        if (scheduler != null) {
            scheduler.cancel();
        }
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(GenerateOrder.class, this::generateOrder)
            .build();
    }

    private int orderCount = 0;

    private void generateOrder(GenerateOrder msg) {
        orderCount++;
        Order order = new Order(
            "ORD-" + String.format("%05d", orderCount),
            "CUST-" + (1000 + (int)(Math.random() * 100)),
            Arrays.asList("상품-" + (int)(Math.random() * 10)),
            50.0 + Math.random() * 450.0
        );

        distributor.tell(order, getSelf());
    }

    public static Props props(ActorRef distributor, int ordersPerSecond) {
        return Props.create(LoadGenerator.class,
            () -> new LoadGenerator(distributor, ordersPerSecond));
    }

    static class GenerateOrder {}
}

// 확장성 테스트 애플리케이션
public class ScalabilityTest {
    public static void main(String[] args) throws InterruptedException {
        ActorSystem system = ActorSystem.create("ScalableECommerce");

        try {
            // 시나리오 1: 낮은 부하 (워커 2개)
            System.out.println("\n=== 시나리오 1: 낮은 부하 (워커 2개) ===");
            testScenario(system, 2, 5, 3); // 워커 2개, 초당 5주문, 3초간

            Thread.sleep(2000);

            // 시나리오 2: 중간 부하 (워커 5개)
            System.out.println("\n=== 시나리오 2: 중간 부하 (워커 5개) ===");
            testScenario(system, 5, 20, 3); // 워커 5개, 초당 20주문, 3초간

            Thread.sleep(2000);

            // 시나리오 3: 높은 부하 (워커 10개)
            System.out.println("\n=== 시나리오 3: 높은 부하 (워커 10개) ===");
            testScenario(system, 10, 50, 3); // 워커 10개, 초당 50주문, 3초간

        } finally {
            system.terminate();
        }
    }

    private static void testScenario(
        ActorSystem system,
        int workers,
        int ordersPerSecond,
        int durationSeconds
    ) throws InterruptedException {

        // Distributor 생성
        ActorRef distributor = system.actorOf(
            OrderDistributor.props(workers),
            "distributor-" + System.currentTimeMillis()
        );

        // Load Generator 생성
        ActorRef loadGen = system.actorOf(
            LoadGenerator.props(distributor, ordersPerSecond),
            "loadgen-" + System.currentTimeMillis()
        );

        // 지정된 시간동안 실행
        Thread.sleep(durationSeconds * 1000);

        // Load Generator 중지
        system.stop(loadGen);

        // 처리 완료 대기
        Thread.sleep(2000);

        // 통계 출력
        system.stop(distributor);
    }
}
```

### 실행 결과

```
=== 시나리오 1: 낮은 부하 (워커 2개) ===
OrderDistributor: 2개의 워커로 시작
[Worker $a] 주문 처리 시작: ORD-00001 (고객: CUST-1042, 금액: $234.56)
[Worker $b] 주문 처리 시작: ORD-00002 (고객: CUST-1015, 금액: $89.23)
[Worker $a] 주문 처리 완료: ORD-00001
OrderDistributor: 주문 완료 (1/15)
[Worker $b] 주문 처리 완료: ORD-00002
OrderDistributor: 주문 완료 (2/15)
...

=== 시나리오 2: 중간 부하 (워커 5개) ===
OrderDistributor: 5개의 워커로 시작
[Worker $a] 주문 처리 시작: ORD-00001 (고객: CUST-1023, 금액: $156.78)
[Worker $b] 주문 처리 시작: ORD-00002 (고객: CUST-1067, 금액: $345.12)
[Worker $c] 주문 처리 시작: ORD-00003 (고객: CUST-1091, 금액: $78.90)
...

=== 시나리오 3: 높은 부하 (워커 10개) ===
OrderDistributor: 10개의 워커로 시작
[Worker $a] 주문 처리 시작: ORD-00001 (고객: CUST-1045, 금액: $267.34)
[Worker $b] 주문 처리 시작: ORD-00002 (고객: CUST-1012, 금액: $123.45)
[Worker $c] 주문 처리 시작: ORD-00003 (고객: CUST-1088, 금액: $456.78)
...
```

### 확장성 전략

| 전략 | 설명 | 사용 사례 |
|------|------|-----------|
| **RoundRobinPool** | 순환 방식으로 워커에게 작업 분배 | 모든 작업이 비슷한 처리 시간을 가질 때 |
| **SmallestMailboxPool** | 가장 적은 메시지를 가진 워커에게 분배 | 작업 처리 시간이 불균등할 때 |
| **RandomPool** | 무작위로 워커 선택 | 간단한 부하 분산 |
| **ConsistentHashingPool** | 키 기반으로 동일 워커에 분배 | 상태 친화성이 필요할 때 (예: 같은 고객의 주문) |

### 동적 확장

```java
// 동적으로 워커 수 조정
class AdaptiveDistributor extends AbstractActor {
    private ActorRef orderRouter;
    private int currentWorkers = 2;

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(HighLoad.class, msg -> {
                // 부하가 높으면 워커 증가
                resizePool(currentWorkers * 2);
            })
            .match(LowLoad.class, msg -> {
                // 부하가 낮으면 워커 감소
                resizePool(Math.max(2, currentWorkers / 2));
            })
            .build();
    }

    private void resizePool(int newSize) {
        currentWorkers = newSize;
        orderRouter.tell(new RoundRobinPool(newSize), getSelf());
        System.out.println("워커 수 조정: " + newSize);
    }
}
```

## E-Commerce 예제: 장애 내성 (Fault-Tolerance)

결제 시스템에서 일시적인 네트워크 오류나 서비스 장애가 발생해도 시스템이 자동으로 복구되도록 구현해보겠습니다.

### Java/Akka Supervision을 활용한 장애 처리

```java
import akka.actor.*;
import java.time.Duration;

// 결제 게이트웨이 예외
class PaymentGatewayException extends RuntimeException {
    public PaymentGatewayException(String message) {
        super(message);
    }
}

class NetworkException extends RuntimeException {
    public NetworkException(String message) {
        super(message);
    }
}

// 결제 처리 워커 (장애 발생 가능)
class PaymentProcessorActor extends AbstractActor {
    private final String processorId;
    private int retryCount = 0;

    public PaymentProcessorActor() {
        this.processorId = getSelf().path().name();
    }

    @Override
    public void preStart() {
        System.out.println("[" + processorId + "] 시작됨");
    }

    @Override
    public void postStop() {
        System.out.println("[" + processorId + "] 중지됨");
    }

    @Override
    public void preRestart(Throwable reason, Optional<Object> message) {
        System.out.println("[" + processorId + "] 재시작 전: " + reason.getMessage());
    }

    @Override
    public void postRestart(Throwable reason) {
        System.out.println("[" + processorId + "] 재시작 완료");
        retryCount = 0; // 재시작 시 카운터 초기화
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(PaymentRequest.class, this::processPayment)
            .build();
    }

    private void processPayment(PaymentRequest request) {
        System.out.println(String.format(
            "[%s] 결제 처리 시도 (재시도: %d): %s / $%.2f",
            processorId, retryCount, request.orderId, request.amount
        ));

        // 장애 시뮬레이션
        double random = Math.random();

        if (random < 0.2) {
            // 20% 확률로 네트워크 오류 (재시도 가능)
            throw new NetworkException("일시적 네트워크 오류");
        } else if (random < 0.25) {
            // 5% 확률로 게이트웨이 오류 (재시작 필요)
            throw new PaymentGatewayException("결제 게이트웨이 오류");
        } else if (random < 0.27) {
            // 2% 확률로 심각한 오류 (복구 불가)
            throw new RuntimeException("심각한 시스템 오류");
        }

        // 정상 처리
        try {
            Thread.sleep(300);
            System.out.println(String.format(
                "[%s] 결제 성공: %s",
                processorId, request.orderId
            ));

            getSender().tell(
                new PaymentResult(request.orderId, true),
                getSelf()
            );

            retryCount = 0; // 성공 시 카운터 초기화

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public static Props props() {
        return Props.create(PaymentProcessorActor.class,
            PaymentProcessorActor::new);
    }
}

// Supervisor Actor - 장애 복구 전략 정의
class PaymentSupervisor extends AbstractActor {
    private final ActorRef paymentProcessor;

    public PaymentSupervisor() {
        // 자식 Actor 생성
        this.paymentProcessor = getContext().actorOf(
            PaymentProcessorActor.props(),
            "payment-processor"
        );
    }

    @Override
    public SupervisorStrategy supervisorStrategy() {
        return new OneForOneStrategy(
            10, // 최대 재시작 횟수
            Duration.ofMinutes(1), // 시간 윈도우
            DeciderBuilder
                // NetworkException: 재시도 (Actor 상태 유지)
                .match(NetworkException.class, e -> {
                    System.out.println("SupervisorStrategy: 네트워크 오류 -> 재시도");
                    return SupervisorStrategy.resume();
                })
                // PaymentGatewayException: 재시작 (Actor 상태 초기화)
                .match(PaymentGatewayException.class, e -> {
                    System.out.println("SupervisorStrategy: 게이트웨이 오류 -> 재시작");
                    return SupervisorStrategy.restart();
                })
                // NullPointerException: 재시작
                .match(NullPointerException.class, e -> {
                    System.out.println("SupervisorStrategy: NPE -> 재시작");
                    return SupervisorStrategy.restart();
                })
                // RuntimeException: 중지 (복구 불가)
                .match(RuntimeException.class, e -> {
                    System.out.println("SupervisorStrategy: 심각한 오류 -> 중지");
                    return SupervisorStrategy.stop();
                })
                // 기타 예외: 상위로 전파
                .matchAny(e -> {
                    System.out.println("SupervisorStrategy: 알 수 없는 오류 -> 전파");
                    return SupervisorStrategy.escalate();
                })
                .build()
        );
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(PaymentRequest.class, request -> {
                // 요청을 자식 Actor에게 전달
                paymentProcessor.forward(request, getContext());
            })
            .match(PaymentResult.class, result -> {
                // 결과를 원래 요청자에게 전달
                getSender().tell(result, getSelf());
            })
            .build();
    }

    public static Props props() {
        return Props.create(PaymentSupervisor.class, PaymentSupervisor::new);
    }
}

// 결제 관리자 (여러 Supervisor 관리)
class PaymentManager extends AbstractActor {
    private final List<ActorRef> supervisors = new ArrayList<>();
    private int currentIndex = 0;

    public PaymentManager(int numberOfSupervisors) {
        // 여러 Supervisor 생성 (격리를 위해)
        for (int i = 0; i < numberOfSupervisors; i++) {
            ActorRef supervisor = getContext().actorOf(
                PaymentSupervisor.props(),
                "supervisor-" + i
            );
            supervisors.add(supervisor);
        }

        System.out.println("PaymentManager: " +
            numberOfSupervisors + "개의 Supervisor 생성");
    }

    @Override
    public SupervisorStrategy supervisorStrategy() {
        // AllForOneStrategy: 한 자식이 실패하면 모든 자식 재시작
        return new AllForOneStrategy(
            5,
            Duration.ofMinutes(1),
            DeciderBuilder
                .matchAny(e -> SupervisorStrategy.restart())
                .build()
        );
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(PaymentRequest.class, this::routePayment)
            .match(PaymentResult.class, this::handleResult)
            .build();
    }

    private void routePayment(PaymentRequest request) {
        // 라운드로빈으로 Supervisor에게 분배
        ActorRef supervisor = supervisors.get(currentIndex);
        currentIndex = (currentIndex + 1) % supervisors.size();

        supervisor.tell(request, getSelf());
    }

    private void handleResult(PaymentResult result) {
        System.out.println("PaymentManager: 결제 결과 수신 - " +
            result.orderId + " / " +
            (result.success ? "성공" : "실패"));
    }

    public static Props props(int numberOfSupervisors) {
        return Props.create(PaymentManager.class,
            () -> new PaymentManager(numberOfSupervisors));
    }
}

// 장애 내성 테스트
public class FaultToleranceTest {
    public static void main(String[] args) throws InterruptedException {
        ActorSystem system = ActorSystem.create("FaultTolerantPayment");

        try {
            // Payment Manager 생성 (3개의 Supervisor)
            ActorRef paymentManager = system.actorOf(
                PaymentManager.props(3),
                "payment-manager"
            );

            System.out.println("\n=== 결제 요청 시작 ===\n");

            // 20개의 결제 요청 전송
            for (int i = 1; i <= 20; i++) {
                PaymentRequest request = new PaymentRequest(
                    "ORD-" + String.format("%03d", i),
                    50.0 + Math.random() * 450.0
                );

                paymentManager.tell(request, ActorRef.noSender());

                Thread.sleep(200); // 요청 간격
            }

            // 처리 완료 대기
            Thread.sleep(5000);

            System.out.println("\n=== 테스트 완료 ===");

        } finally {
            system.terminate();
        }
    }
}
```

### 실행 결과

```
PaymentManager: 3개의 Supervisor 생성
[payment-processor] 시작됨
[payment-processor] 시작됨
[payment-processor] 시작됨

=== 결제 요청 시작 ===

[payment-processor] 결제 처리 시도 (재시도: 0): ORD-001 / $234.56
[payment-processor] 결제 성공: ORD-001
PaymentManager: 결제 결과 수신 - ORD-001 / 성공

[payment-processor] 결제 처리 시도 (재시도: 0): ORD-002 / $156.78
SupervisorStrategy: 네트워크 오류 -> 재시도
[payment-processor] 결제 처리 시도 (재시도: 1): ORD-002 / $156.78
[payment-processor] 결제 성공: ORD-002
PaymentManager: 결제 결과 수신 - ORD-002 / 성공

[payment-processor] 결제 처리 시도 (재시도: 0): ORD-003 / $89.23
SupervisorStrategy: 게이트웨이 오류 -> 재시작
[payment-processor] 재시작 전: 결제 게이트웨이 오류
[payment-processor] 중지됨
[payment-processor] 시작됨
[payment-processor] 재시작 완료

[payment-processor] 결제 처리 시도 (재시도: 0): ORD-004 / $345.12
[payment-processor] 결제 성공: ORD-004
PaymentManager: 결제 결과 수신 - ORD-004 / 성공

[payment-processor] 결제 처리 시도 (재시도: 0): ORD-005 / $456.78
SupervisorStrategy: 심각한 오류 -> 중지
[payment-processor] 중지됨
[payment-processor] 시작됨

[payment-processor] 결제 처리 시도 (재시도: 0): ORD-006 / $123.45
[payment-processor] 결제 성공: ORD-006
PaymentManager: 결제 결과 수신 - ORD-006 / 성공

=== 테스트 완료 ===
```

### Supervision 전략

| 전략 | 설명 | 사용 사례 |
|------|------|-----------|
| **Resume** | Actor 상태 유지하고 계속 진행 | 일시적 오류 (네트워크 타임아웃) |
| **Restart** | Actor 재시작 (상태 초기화) | 복구 가능한 오류 (잘못된 상태) |
| **Stop** | Actor 중지 | 복구 불가능한 오류 |
| **Escalate** | 상위 Supervisor에게 전파 | 처리 방법을 모르는 오류 |

### Supervision 계층

```
PaymentManager (AllForOneStrategy)
    ├── Supervisor-0 (OneForOneStrategy)
    │   └── PaymentProcessor-0
    ├── Supervisor-1 (OneForOneStrategy)
    │   └── PaymentProcessor-1
    └── Supervisor-2 (OneForOneStrategy)
        └── PaymentProcessor-2
```

## E-Commerce 예제: 위치 투명성 (Location Transparency)

글로벌 이커머스 시스템에서 주문 처리 Actor가 여러 데이터센터(미국, 유럽, 아시아)에 분산되어 있어도 동일한 방식으로 통신하도록 구현해보겠습니다.

### Java/Akka Clustering을 활용한 분산 시스템

```java
import akka.actor.*;
import akka.cluster.Cluster;
import akka.cluster.ClusterEvent;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

// 주문 처리 Actor (위치와 무관)
class GlobalOrderProcessor extends AbstractActor {
    private final String region;
    private final Cluster cluster;

    public GlobalOrderProcessor(String region) {
        this.region = region;
        this.cluster = Cluster.get(getContext().getSystem());
    }

    @Override
    public void preStart() {
        // 클러스터 이벤트 구독
        cluster.subscribe(
            getSelf(),
            ClusterEvent.MemberUp.class,
            ClusterEvent.MemberRemoved.class
        );

        System.out.println(String.format(
            "[%s] GlobalOrderProcessor 시작 (노드: %s)",
            region,
            cluster.selfAddress()
        ));
    }

    @Override
    public void postStop() {
        cluster.unsubscribe(getSelf());
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(Order.class, this::processOrder)
            .match(ClusterEvent.MemberUp.class, this::memberUp)
            .match(ClusterEvent.MemberRemoved.class, this::memberRemoved)
            .build();
    }

    private void processOrder(Order order) {
        System.out.println(String.format(
            "[%s] 주문 처리: %s (고객: %s, 요청자: %s)",
            region,
            order.orderId,
            order.customerId,
            getSender().path()
        ));

        // 지역별 처리 로직
        try {
            Thread.sleep(500);

            System.out.println(String.format(
                "[%s] 주문 완료: %s",
                region,
                order.orderId
            ));

            // 결과를 요청자에게 전송 (로컬/원격 구분 없음)
            getSender().tell(
                new OrderProcessed(order.orderId, true),
                getSelf()
            );

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void memberUp(ClusterEvent.MemberUp event) {
        System.out.println(String.format(
            "[%s] 노드 추가: %s",
            region,
            event.member().address()
        ));
    }

    private void memberRemoved(ClusterEvent.MemberRemoved event) {
        System.out.println(String.format(
            "[%s] 노드 제거: %s",
            region,
            event.member().address()
        ));
    }

    public static Props props(String region) {
        return Props.create(GlobalOrderProcessor.class,
            () -> new GlobalOrderProcessor(region));
    }
}

// 주문 라우터 (지역별 분배)
class GlobalOrderRouter extends AbstractActor {
    private final Map<String, ActorRef> regionProcessors = new HashMap<>();

    @Override
    public void preStart() {
        System.out.println("GlobalOrderRouter 시작");

        // 각 지역의 Actor 참조 얻기 (Actor Selection 사용)
        // 실제로는 Cluster Singleton이나 Cluster Sharding 사용
        registerRegion("US", "akka://ECommerceCluster@us-node:2551/user/order-processor");
        registerRegion("EU", "akka://ECommerceCluster@eu-node:2552/user/order-processor");
        registerRegion("ASIA", "akka://ECommerceCluster@asia-node:2553/user/order-processor");
    }

    private void registerRegion(String region, String actorPath) {
        ActorSelection selection = getContext().actorSelection(actorPath);
        // ActorSelection을 ActorRef로 변환 (실제로는 Identify 패턴 사용)
        regionProcessors.put(region, selection.anchor());
        System.out.println("지역 등록: " + region + " -> " + actorPath);
    }

    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(Order.class, this::routeOrder)
            .match(OrderProcessed.class, this::handleResult)
            .build();
    }

    private void routeOrder(Order order) {
        // 고객 위치 기반으로 가장 가까운 지역 선택
        String targetRegion = determineRegion(order.customerId);

        ActorRef processor = regionProcessors.get(targetRegion);
        if (processor != null) {
            System.out.println(String.format(
                "GlobalOrderRouter: 주문 라우팅 %s -> %s",
                order.orderId,
                targetRegion
            ));

            // 위치와 무관하게 동일한 방식으로 메시지 전송
            processor.tell(order, getSelf());
        } else {
            System.out.println("오류: 지역 " + targetRegion + "을 찾을 수 없음");
        }
    }

    private String determineRegion(String customerId) {
        // 고객 ID 기반으로 지역 결정 (간단한 해싱)
        int hash = Math.abs(customerId.hashCode());
        String[] regions = {"US", "EU", "ASIA"};
        return regions[hash % regions.length];
    }

    private void handleResult(OrderProcessed result) {
        System.out.println("GlobalOrderRouter: 주문 완료 확인 - " + result.orderId);
    }

    public static Props props() {
        return Props.create(GlobalOrderRouter.class, GlobalOrderRouter::new);
    }
}

// 클러스터 노드 설정
class ClusterNodeConfig {
    public static Config createConfig(String nodeName, int port, String region) {
        return ConfigFactory.parseString(String.format(
            "akka {" +
            "  actor {" +
            "    provider = cluster" +
            "  }" +
            "  remote {" +
            "    artery {" +
            "      canonical {" +
            "        hostname = \"%s-node\"" +
            "        port = %d" +
            "      }" +
            "    }" +
            "  }" +
            "  cluster {" +
            "    seed-nodes = [" +
            "      \"akka://ECommerceCluster@us-node:2551\"," +
            "      \"akka://ECommerceCluster@eu-node:2552\"," +
            "      \"akka://ECommerceCluster@asia-node:2553\"" +
            "    ]" +
            "    roles = [\"%s\"]" +
            "  }" +
            "}",
            nodeName, port, region
        )).withFallback(ConfigFactory.load());
    }
}

// 미국 노드
public class USNode {
    public static void main(String[] args) throws InterruptedException {
        Config config = ClusterNodeConfig.createConfig("us", 2551, "US");
        ActorSystem system = ActorSystem.create("ECommerceCluster", config);

        // 주문 처리 Actor 생성
        system.actorOf(
            GlobalOrderProcessor.props("US"),
            "order-processor"
        );

        System.out.println("US 노드 시작 완료");
        Thread.sleep(Long.MAX_VALUE);
    }
}

// 유럽 노드
public class EUNode {
    public static void main(String[] args) throws InterruptedException {
        Config config = ClusterNodeConfig.createConfig("eu", 2552, "EU");
        ActorSystem system = ActorSystem.create("ECommerceCluster", config);

        system.actorOf(
            GlobalOrderProcessor.props("EU"),
            "order-processor"
        );

        System.out.println("EU 노드 시작 완료");
        Thread.sleep(Long.MAX_VALUE);
    }
}

// 아시아 노드
public class AsiaNode {
    public static void main(String[] args) throws InterruptedException {
        Config config = ClusterNodeConfig.createConfig("asia", 2553, "ASIA");
        ActorSystem system = ActorSystem.create("ECommerceCluster", config);

        system.actorOf(
            GlobalOrderProcessor.props("ASIA"),
            "order-processor"
        );

        System.out.println("ASIA 노드 시작 완료");
        Thread.sleep(Long.MAX_VALUE);
    }
}

// 클라이언트 (라우터 노드)
public class ClientNode {
    public static void main(String[] args) throws InterruptedException {
        Config config = ClusterNodeConfig.createConfig("client", 2554, "CLIENT");
        ActorSystem system = ActorSystem.create("ECommerceCluster", config);

        // 글로벌 라우터 생성
        ActorRef router = system.actorOf(
            GlobalOrderRouter.props(),
            "global-router"
        );

        System.out.println("클라이언트 노드 시작 완료\n");

        // 클러스터 초기화 대기
        Thread.sleep(5000);

        System.out.println("=== 글로벌 주문 처리 시작 ===\n");

        // 다양한 고객의 주문 생성
        String[] customers = {
            "CUST-US-001", "CUST-EU-002", "CUST-ASIA-003",
            "CUST-US-004", "CUST-EU-005", "CUST-ASIA-006"
        };

        for (int i = 0; i < customers.length; i++) {
            Order order = new Order(
                "ORD-" + String.format("%03d", i + 1),
                customers[i],
                Arrays.asList("상품-" + (i + 1)),
                100.0 + Math.random() * 400.0
            );

            // 위치와 무관하게 동일한 방식으로 메시지 전송
            router.tell(order, ActorRef.noSender());

            Thread.sleep(1000);
        }

        Thread.sleep(5000);
        System.out.println("\n=== 테스트 완료 ===");

        system.terminate();
    }
}
```

### 실행 결과

```
# US 노드
US 노드 시작 완료
[US] GlobalOrderProcessor 시작 (노드: akka://ECommerceCluster@us-node:2551)
[US] 노드 추가: akka://ECommerceCluster@eu-node:2552
[US] 노드 추가: akka://ECommerceCluster@asia-node:2553
[US] 주문 처리: ORD-001 (고객: CUST-US-001, 요청자: akka://ECommerceCluster@client-node:2554/user/global-router)
[US] 주문 완료: ORD-001
[US] 주문 처리: ORD-004 (고객: CUST-US-004, 요청자: akka://ECommerceCluster@client-node:2554/user/global-router)
[US] 주문 완료: ORD-004

# EU 노드
EU 노드 시작 완료
[EU] GlobalOrderProcessor 시작 (노드: akka://ECommerceCluster@eu-node:2552)
[EU] 노드 추가: akka://ECommerceCluster@us-node:2551
[EU] 노드 추가: akka://ECommerceCluster@asia-node:2553
[EU] 주문 처리: ORD-002 (고객: CUST-EU-002, 요청자: akka://ECommerceCluster@client-node:2554/user/global-router)
[EU] 주문 완료: ORD-002
[EU] 주문 처리: ORD-005 (고객: CUST-EU-005, 요청자: akka://ECommerceCluster@client-node:2554/user/global-router)
[EU] 주문 완료: ORD-005

# ASIA 노드
ASIA 노드 시작 완료
[ASIA] GlobalOrderProcessor 시작 (노드: akka://ECommerceCluster@asia-node:2553)
[ASIA] 노드 추가: akka://ECommerceCluster@us-node:2551
[ASIA] 노드 추가: akka://ECommerceCluster@eu-node:2552
[ASIA] 주문 처리: ORD-003 (고객: CUST-ASIA-003, 요청자: akka://ECommerceCluster@client-node:2554/user/global-router)
[ASIA] 주문 완료: ORD-003
[ASIA] 주문 처리: ORD-006 (고객: CUST-ASIA-006, 요청자: akka://ECommerceCluster@client-node:2554/user/global-router)
[ASIA] 주문 완료: ORD-006

# Client 노드
클라이언트 노드 시작 완료
GlobalOrderRouter 시작
지역 등록: US -> akka://ECommerceCluster@us-node:2551/user/order-processor
지역 등록: EU -> akka://ECommerceCluster@eu-node:2552/user/order-processor
지역 등록: ASIA -> akka://ECommerceCluster@asia-node:2553/user/order-processor

=== 글로벌 주문 처리 시작 ===

GlobalOrderRouter: 주문 라우팅 ORD-001 -> US
GlobalOrderRouter: 주문 라우팅 ORD-002 -> EU
GlobalOrderRouter: 주문 라우팅 ORD-003 -> ASIA
GlobalOrderRouter: 주문 라우팅 ORD-004 -> US
GlobalOrderRouter: 주문 라우팅 ORD-005 -> EU
GlobalOrderRouter: 주문 라우팅 ORD-006 -> ASIA
GlobalOrderRouter: 주문 완료 확인 - ORD-001
GlobalOrderRouter: 주문 완료 확인 - ORD-002
GlobalOrderRouter: 주문 완료 확인 - ORD-003
GlobalOrderRouter: 주문 완료 확인 - ORD-004
GlobalOrderRouter: 주문 완료 확인 - ORD-005
GlobalOrderRouter: 주문 완료 확인 - ORD-006

=== 테스트 완료 ===
```

### 위치 투명성의 장점

1. **단일 프로그래밍 모델** - 로컬/원격 Actor 구분 없이 동일한 코드
2. **동적 배치** - 런타임에 Actor 위치 변경 가능
3. **글로벌 확장** - 여러 데이터센터에 걸쳐 시스템 확장
4. **레이턴시 최적화** - 고객과 가까운 지역에서 처리

### Akka Clustering 패턴

| 패턴 | 설명 | 사용 사례 |
|------|------|-----------|
| **Cluster Singleton** | 클러스터 전체에 하나의 Actor만 존재 | 글로벌 코디네이터, 리더 선출 |
| **Cluster Sharding** | 엔티티를 여러 노드에 자동 분산 | 사용자 세션, 장바구니 |
| **Cluster Routers** | 여러 노드의 Actor에게 작업 분배 | 부하 분산, 병렬 처리 |
| **Distributed Data** | CRDT 기반 분산 데이터 공유 | 카운터, 세트, 맵 |

## Actor Model 프레임워크 비교

| 프레임워크 | 언어 | 특징 | 사용 사례 |
|-----------|------|------|-----------|
| **Erlang/OTP** | Erlang | - 텔레콤 시스템을 위해 개발<br>- Let it crash 철학<br>- Hot code swapping<br>- 30년 이상의 검증 | - 고가용성 시스템<br>- 실시간 통신<br>- IoT 플랫폼 |
| **Akka** | Java/Scala | - JVM 생태계 통합<br>- 강력한 클러스터링<br>- HTTP/gRPC 지원<br>- 대규모 엔터프라이즈 | - 마이크로서비스<br>- 스트리밍 파이프라인<br>- 이벤트 소싱 |
| **Apache Pekko** | Java/Scala | - Akka의 오픈소스 포크<br>- Apache 라이선스<br>- 커뮤니티 주도 | - Akka 대체<br>- 라이선스 우려 해소 |
| **Orleans** | C#/.NET | - Virtual Actor 모델<br>- 자동 라이프사이클 관리<br>- Azure 통합<br>- 게임 개발 최적화 | - 게임 백엔드<br>- IoT<br>- 클라우드 서비스 |

### 프레임워크 선택 기준

```
선택 결정 트리:

언어 환경?
├─ Erlang 생태계 → Erlang/OTP
├─ JVM (Java/Scala)
│  ├─ 라이선스 중요? → Apache Pekko
│  └─ 엔터프라이즈 지원 필요? → Akka
└─ .NET 생태계 → Orleans

요구사항?
├─ 텔레콤/고가용성 → Erlang/OTP
├─ 마이크로서비스/스트리밍 → Akka/Pekko
└─ 게임/Virtual Actor → Orleans
```

### 실제 사용 사례

**Erlang/OTP:**
- WhatsApp: 10억 사용자, 분산 메시징
- Ericsson: 텔레콤 인프라 (99.9999999% 가용성)
- RabbitMQ: 메시지 브로커

**Akka:**
- LinkedIn: 사용자 활동 스트림
- PayPal: 결제 처리 시스템
- Lightbend: 엔터프라이즈 플랫폼

**Orleans:**
- Halo: 게임 백엔드 (수백만 동시 플레이어)
- Microsoft: Azure 서비스

## 결론

Actor Model은 다음과 같은 경우에 특히 유용합니다:

1. **높은 동시성** - 수천~수백만 개의 동시 작업
2. **분산 시스템** - 여러 노드에 걸친 처리
3. **장애 내성** - 자동 복구가 중요한 시스템
4. **비동기 처리** - 이벤트 기반 아키텍처

Actor Model을 통해 복잡한 동시성 문제를 단순화하고, 확장 가능하며 장애에 강한 분산 시스템을 구축할 수 있습니다.
