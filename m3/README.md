# Abstract

M3 is a cluster for long-term logging solution.

# Materials

* [An introduction to M3](https://aiven.io/blog/an-introduction-to-m3)
* [M3 and Prometheus, Monitoring at Planet Scale for Everyone - Rob Skillington, Uber @ youtube](https://www.youtube.com/watch?v=EFutyuIpFXQ)
  * [pdf](https://static.sched.com/hosted_files/kccnceu19/e0/M3%20and%20Prometheus%2C%20Monitoring%20at%20Planet%20Scale%20for%20Everyone.pdf)
  * [event](https://kccnceu19.sched.com/event/MPbX)
* [M3 Documentation](https://m3db.io/docs/)

# Basic

## Creating a Single Node M3DB Cluster with Docker

* [Creating a Single Node M3DB Cluster with Docker](https://m3db.io/docs/quickstart/docker/)

```bash
$ docker run --rm -d -p 7201:7201 -p 7203:7203 --name m3db -v $(pwd)/m3db_data:/var/lib/m3db quay.io/m3db/m3dbnode:latest
```

## M3DB Cluster Deployment, Manually

* [m3 stack](https://github.com/m3db/m3/tree/master/scripts/development/m3_stack)

```bash
$ git clone https://github.com/m3db/m3.git
$ cd m3/scripts/development/m3_stack/
$ chmod 644 prometheus.yml
$ ./start_m3.sh
$ ./stop_m3.sh
# Open browser with xxx.xxx.xxx.xxx:3000 for grafana
# Open browser with xxx.xxx.xxx.xxx:9090 for prometheus
```

## setting up M3DB on Kubernetes

* [m3db-operator @ github](https://github.com/m3db/m3db-operator)



