- [Abstract](#abstract)
- [Materials](#materials)
- [Deploy](#deploy)
- [Architecture](#architecture)

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
