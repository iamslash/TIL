# Abstract

consul 은 다음과 같은 기능을 제공하는 application 이다. 주로 Service Discovery 에 사용한다.

* Service Discovery
* Health Checking
* Key/Value Store
* Multi Datacenter
* Service Segmentation

# Materials

* [consul getting started @ hashicorp](https://learn.hashicorp.com/consul)
* [Consul @ joinc](https://www.joinc.co.kr/w/man/12/consul)
* [Consul @ github](https://github.com/hashicorp/consul)

# Install

## Install 

* [Download Consul](https://www.consul.io/downloads.html)

Download and unzip and move to `/usr/local/bin`.

## Install With Docker

* [Consul with Containers](https://learn.hashicorp.com/tutorials/consul/docker-container-agents)

```bash
# Get the Docker image
$ docker pull consul
$ docker images -f 'reference=consul'

# Configure and run agent as a Consul Server
$ docker run --rm -d -p 8500:8500 -p 8600:8600/udp --name=badger consul agent -server -ui -node=server-1 -bootstrap-expect=1 -client=0.0.0.0

# Discover the server IP address
$ docker exec badger consul members
Node      Address          Status  Type    Build  Protocol  DC   Segment
server-1  172.17.0.2:8301  alive   server  1.8.4  2         dc1  <all>
```

```bash
# Configure and run agent as a Consul client
$ docker run --rm --name=fox consul agent -node=client-1 -join=172.17.0.2

$ docker exec badger consul members
server-1  172.17.0.2:8301  alive   server  1.8.4  2         dc1  <all>
client-1  172.17.0.3:8301  alive   client  1.8.4  2         dc1  <default>
```

```bash
# Register a service
$ docker pull hashicorp/counting-service:0.0.2
$ docker run --rm -p 9001:9001 -d --name=weasel hashicorp/counting-service:0.0.2
$ docker exec fox /bin/sh -c "echo '{\"service\": {\"name\": \"counting\", \"tags\": [\"go\"], \"port\": 9001}}' >> /consul/config/counting.json"
$ docker exec fox consul reload
```

```bash
# Use Consul DNS to discover the counting service
$ dig @127.0.0.1 -p 8600 counting.service.consul
# Open browser http://localhost:8500
```

# Basic

## Start, Stop consul

하나의 `consul` 은 server 혹은 client 로 실행 가능하다. server 는 최소 1 대 있어야 하고 보통 3 혹은 5 대를 운영한다.
client 는 service application 마다 하나씩 있어야 한다.

client 는 service 를 등록, health check, query 를 server 에 전달한다.

```bash
# Run consul server with dev mode
$ consul agent -dev

# Check consul servers in same datacenter
$ consul members
$ consul members -detailed

# Request HTTP to local consul server.
$ curl localhost:8500/v1/catalog/nodes
[
    {
        "ID": "019063f6-9215-6f2c-c930-9e84600029da",
        "Node": "Judiths-MBP",
        "Address": "127.0.0.1",
        "Datacenter": "dc1",
        "TaggedAddresses": {
            "lan": "127.0.0.1",
            "wan": "127.0.0.1"
        },
        "Meta": {
            "consul-network-segment": ""
        },
        "CreateIndex": 9,
        "ModifyIndex": 10
    }
]

# Consul server has DNS interface.
# Let's DNS query with dig command.
$ dig @127.0.0.1 -p 8600 Judiths-MBP.node.consul

# Stop the consul server (agent) gracefully.
$ consul leave
```

## Register a Service and Health Check - Service Discovery

service 는 Consul configuration file 혹은 HTTP API 로 등록할 수 있다. Consul configuration 은 Consul server (agent) 가 실행할 때 읽는다. Consul client 가 Consul configuration file 을 읽어서 Consul server 에게 전달할 수 있는가???

service 가 Consul catalog 에 등록되면 DNS interface 혹은 HTTP API 로 query 가 가능하다.

```bash
$ mkdir ./consul.d

$ echo '{"service":
  {"name": "web",
   "tags": ["rails"],
   "port": 80
  }
}' > ./consul.d/web.json

# 
$ consul agent -dev -enable-script-checks -config-dir=./consul.d

# The fully-qualified domain name of the web service is web.service.consul. 
# Query the DNS interface (which Consul runs by default on port 8600) for the registered service.
$ dig @127.0.0.1 -p 8600 web.service.consul

; <<>> DiG 9.10.6 <<>> @127.0.0.1 -p 8600 web.service.consul
; (1 server found)
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 28340
;; flags: qr aa rd; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 2
;; WARNING: recursion requested but not available

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 4096
;; QUESTION SECTION:
;web.service.consul.        IN  A

;; ANSWER SECTION:
web.service.consul. 0   IN  A   127.0.0.1

;; ADDITIONAL SECTION:
web.service.consul. 0   IN  TXT "consul-network-segment="

;; Query time: 2 msec
;; SERVER: 127.0.0.1#8600(127.0.0.1)
;; WHEN: Tue Jul 16 14:26:53 PDT 2019
;; MSG SIZE  rcvd: 99

# As you can see, an A record was returned containing the IP address where the service was registered. A records can only hold IP addresses.

# You can also use the DNS interface to retrieve the entire address/port pair as a SRV record.
$ dig @127.0.0.1 -p 8600 web.service.consul SRV
# The SRV record says that the web service is running on port 80 and exists on the node Judiths-MBP.lan.node.dc1.consul.. 
# An additional section is returned by the DNS with the A record for that node.
```

```bash
# query with HTTP API
$ curl http://localhost:8500/v1/catalog/service/web

# Query with HTTP API, only health services.
$ curl 'http://localhost:8500/v1/health/service/web?passing'

# Let's add health check port
$ echo '{"service":
  {"name": "web",
    "tags": ["rails"],
    "port": 80,
    "check": {
      "args": ["curl", "localhost"],
      "interval": "10s"
    }
  }
}' > ./consul.d/web.json

# There is no downtime.
$ consul reload

# Query again
$ dig @127.0.0.1 -p 8600 web.service.consul
```

## Connect Services - Service Mesh

Consul 에 등록된 service 들은 consul sidecar proxy 를 통해 서로 연결될 수 있다. 이것을 Consul Connect 라고 한다.

Consul sidecar proxy 는 TLS 를 통해 서로 통신한다. service to service traffic 을 monitoring 하고 prometheus 를 위한 metric data 를 제공한다.

Consul Connect 는 다음과 같은 방법으로 실행한다.

* Start a service.
* Register it normally, but with an additional connect stanza.
* Register a second proxy to communicate with the service.
* Start sidecar proxies.
* Practice blocking the connection to the service by creating an intention.


```bash
# Start the socat service and specify that it will listen for TCP connections on port 8181.
$ socat -v tcp-l:8181,fork exec:"/bin/cat"

$ nc 127.0.0.1 8181
hello
hello
testing 123
testing 123

# Register the Service and Proxy with Consul
$ echo '{
  "service": {
    "name": "socat",
    "port": 8181,
    "connect": { "sidecar_service": {} }
  }
}' > ./consul.d/socat.json

$ consul connect proxy -sidecar-for socat

# Register a Dependent Service and Proxy
$ echo '{"service": {
    "name": "web",
    "port": 8080,
    "connect": {
      "sidecar_service": {
        "proxy": {
          "upstreams": [{
             "destination_name": "socat",
             "local_bind_port": 9191
          }]
        }
      }
    }
  }
}' > ./consul.d/web.json

$ nc 127.0.0.1 9191

# Now start the web proxy using the configuration from the sidecar registration.
$ consul connect proxy -sidecar-for web

# Try connecting to socat again on port 9191. This time it should work and echo back your text.
$ nc 127.0.0.1 9191
hello
hello
```

또한 intension 을 생성하여 서비스별 통신을 제한할 수 있다.

```bash
# Create an intention to deny access from web to socat that specifies policy, and the source and destination services.
$ consul intention create -deny web socat

# Now, try to connect again. The connection will fail.
$ nc 127.0.0.1 9191

# Delete the intention.
$ consul intention delete web socat

# Try the connection again, and it will succeed.
$ nc 127.0.0.1 9191
```

# Prometheus metrics

* [View Metrics](https://www.consul.io/api/agent.html#view-metrics)

```bash
$ curl \
    http://127.0.0.1:8500/v1/agent/metrics
```
