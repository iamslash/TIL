# Abstract

Application 을 load test 하는 방법에 대해 적는다.

# Prerequites

* System 의 Endpoint 들을 파악한다. (MySQL, Redis, Kafka, etc...)
* Metric Monitoring 을 준비한다. (Grafana Dash Board)
* Stress Test Client 를 준비한다. (ngrinder, k6, etc...)
* Stress Test Client Parameter 들을 준비한다. (process count, thread count, etc...)
* Application Parameter 들을 준비한다. (max threads, mysql max pool connections, CPU, RAM, etc...)
* 결과를 기록할 양식을 준비한다. (Excel)

# Work Flow

* Application 의 CPU 를 80% 이상 사용하는 Stress Test Client Parameter 들을 찾는다. 
* Application Parameter, Stree Test Client Parameter 들을 바꿔가면서 Stress Test 를 한다. 그리고 TPS 를 기록한다.
* Application Parameter 의 최적 option 을 찾는다.
