- [Abstract](#abstract)
- [Materials](#materials)
- [Deploy](#deploy)
- [Architecture](#architecture)
- [Example: Money Transfer](#example-money-transfer)
- [Example: Subscription Workflow](#example-subscription-workflow)
- [Example: SAGA](#example-saga)
- [Load Tester](#load-tester)

---

# Abstract

Temporal is a task orchestration framework. batch job 들의 상태를 관리하고 실행한다. 

# Materials

* [temporalio | github](https://github.com/temporalio/temporal)
  * [sdk-go](https://github.com/temporalio/sdk-go)
  * [samples-go](https://github.com/temporalio/samples-go)
* [cadence | github](https://github.com/uber/cadence)
  * uber 에서 만든 task orchestation framework 이다. temporal 의 조상임.
* [Build an eCommerce App With Temporal, Part 1: Getting Started](https://docs.temporal.io/blog/build-an-ecommerce-app-with-temporal-part-1/)
  * [src](https://github.com/temporalio/temporal-ecommerce)
  * [Building an eCommerce web app with Temporal, Part 2: Reminder Emails](https://docs.temporal.io/blog/build-an-ecommerce-app-with-temporal-part-2-reminder-emails/)
  * [Building an eCommerce web app with Temporal, Part 3: Testing](https://docs.temporal.io/blog/build-an-ecommerce-app-with-temporal-part-3-testing/)
  * [Building an eCommerce web app with Temporal, Part 4: REST API](https://docs.temporal.io/blog/build-an-ecommerce-app-with-temporal-part-4-rest-api/)

# Deploy

Docker Runtime 의 memory 가 충분히 커야함. (16GB) 그렇지 않으면 Elasticsearch 가 실행되지 못함.x

```bash
git clone https://github.com/temporalio/docker-compose.git
cd  docker-compose
docker-compose up
# Open the browser for admin console. http://localhost:8088/
```

# Architecture

> [Temporal Server](https://docs.temporal.io/docs/server/production-deployment#temporal-server)

---

Temporal Server is consisted of these 4 services

* Frontend gateway: for rate limiting, routing, authorizing
* History subsystem: maintains data (mutable state, queues, and timers)
* Matching subsystem: hosts Task Queues for dispatching
* Worker service: for internal background workflows

# Example: Money Transfer

* [money-transfer-project-template-go](https://github.com/temporalio/money-transfer-project-template-go)

`Withdraw()` 는 성공하고 `Deposit()` 는 실패했다고 해보자. 다음번에
`TransferMoney()` 는 다시 실행될테고 `Withdraw()` 는 건너뛰고 `Deposit()` 이
다시 실행된다.

```go
func TransferMoney(ctx workflow.Context, transferDetails TransferDetails) error {
	// RetryPolicy specifies how to automatically handle retries if an Activity fails.
	retrypolicy := &temporal.RetryPolicy{
		InitialInterval:    time.Second,
		BackoffCoefficient: 2.0,
		MaximumInterval:    time.Minute,
		MaximumAttempts:    500,
	}
	options := workflow.ActivityOptions{
		// Timeout options specify when to automatically timeout Activity functions.
		StartToCloseTimeout: time.Minute,
		// Optionally provide a customized RetryPolicy.
		// Temporal retries failures by default, this is just an example.
		RetryPolicy: retrypolicy,
	}
	ctx = workflow.WithActivityOptions(ctx, options)
	err := workflow.ExecuteActivity(ctx, Withdraw, transferDetails).Get(ctx, nil)
	if err != nil {
		return err
	}
	err = workflow.ExecuteActivity(ctx, Deposit, transferDetails).Get(ctx, nil)
	if err != nil {
		return err
	}
	return nil
}
```

# Example: Subscription Workflow

* [subscription-workflow-project-template-go](https://github.com/temporalio/subscription-workflow-project-template-go)

# Example: SAGA

* [TripBookingWorflow](https://github.dev/temporalio/samples-java/blob/079316d43d23bad3b785a3b1fa38f0c572321a28/src/main/java/io/temporal/samples/bookingsaga/TripBookingWorkflowImpl.java#L1)

`addCompensation()` 으로 주욱 매달아 놓으면 끝이다.

```java
public class TripBookingWorkflowImpl implements TripBookingWorkflow {

  private final ActivityOptions options =
      ActivityOptions.newBuilder()
          .setStartToCloseTimeout(Duration.ofHours(1))
          // disable retries for example to run faster
          .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
          .build();
  private final TripBookingActivities activities =
      Workflow.newActivityStub(TripBookingActivities.class, options);

  @Override
  public void bookTrip(String name) {
    // Configure SAGA to run compensation activities in parallel
    Saga.Options sagaOptions = new Saga.Options.Builder().setParallelCompensation(true).build();
    Saga saga = new Saga(sagaOptions);
    try {
      String carReservationID = activities.reserveCar(name);
      saga.addCompensation(activities::cancelCar, carReservationID, name);

      String hotelReservationID = activities.bookHotel(name);
      saga.addCompensation(activities::cancelHotel, hotelReservationID, name);

      String flightReservationID = activities.bookFlight(name);
      saga.addCompensation(activities::cancelFlight, flightReservationID, name);
    } catch (ActivityFailure e) {
      saga.compensate();
      throw e;
    }
  }
}
```

# Load Tester

* [Maru: Load Testing Tool for Temporal Workflows](https://mikhail.io/2021/03/maru-load-testing-tool-for-temporal-workflows/)

---

[maru](https://github.com/temporalio/maru/) 로 load testing 할 수 있음.
