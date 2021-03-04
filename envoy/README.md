- [Materials](#materials)
- [Terms](#terms)
- [Install with Docker](#install-with-docker)
  - [Run Envoy](#run-envoy)
  - [Inspect Configuration](#inspect-configuration)
  - [Update Configuration for "Empty reply from server" error when connect to 9901](#update-configuration-for-empty-reply-from-server-error-when-connect-to-9901)
- [Advanced](#advanced)
  - [transcoding between grpc, http](#transcoding-between-grpc-http)

----

# Materials

* [debugging envoy on macos](https://medium.com/@dirao/debugging-envoyproxy-on-osx-a3ebe87dc916)

# Terms

| Terms | Description |
|---|---|
| Downstream | Client host of Envoy |
| Upstream | Server host of Envoy |
| Listener | Downstream host 의 요청을 받는 부분 |
| Cluster | Upstream host 의 그룹 |

# Install with Docker

## Run Envoy

* [envoy proxy란? (basic)](https://gruuuuu.github.io/cloud/envoy-proxy/#)

----

```console
$ docker pull envoyproxy/envoy-dev
$ docker run --rm -d -p 9901:9901 -p 10000:10000 envoyproxy/envoy-dev
$ curl -v localhost:10000
```

## Inspect Configuration

```console
$ docker exec -it 8f788d9f5bfa /bin/bash
$ cat /etc/envoy/envoy.yaml
```

```yaml
admin:
  access_log_path: /tmp/admin_access.log
  address:
    socket_address:
      protocol: TCP
      address: 127.0.0.1
      port_value: 9901
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        protocol: TCP
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.config.filter.network.http_connection_manager.v2.HttpConnectionManager
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  host_rewrite: www.google.com
                  cluster: service_google
          http_filters:
          - name: envoy.filters.http.router
  clusters:
  - name: service_google
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    # Comment out the following line to test on v6 networks
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: service_google
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: www.google.com
                port_value: 443
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.api.v2.auth.UpstreamTlsContext
        sni: www.google.com
```

## Update Configuration for "Empty reply from server" error when connect to 9901

```console
$ mkdir envoy-mod
$ cd envoy-mod
$ vim envoy.yaml
$ vim Dockerfile
$ docker build -t envoy-mod:v1 .
$ docker run -d --rm --name my-envoy -p 9901:9901 -p 10001:10001 envoy-mod:v1
```

* Dockerfile

```Dockerfile
FROM envoyproxy/envoy-dev 
COPY envoy.yaml /etc/envoy/envoy.yaml
```

* envoy.yaml

  * Update `127.0.0.1` to `0.0.0.0`

```yaml
admin:
  access_log_path: /tmp/admin_access.log
  address:
    socket_address:
      protocol: TCP
      address: 0.0.0.0
      port_value: 9901
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        protocol: TCP
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.config.filter.network.http_connection_manager.v2.HttpConnectionManager
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  host_rewrite: www.google.com
                  cluster: service_google
          http_filters:
          - name: envoy.filters.http.router
  clusters:
  - name: service_google
    connect_timeout: 0.25s
    type: LOGICAL_DNS
    # Comment out the following line to test on v6 networks
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: service_google
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: www.google.com
                port_value: 443
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.api.v2.auth.UpstreamTlsContext
        sni: www.google.com

```

# Advanced

## transcoding between grpc, http

> * [Transcoding gRPC to HTTP/JSON using Envoy](https://blog.jdriven.com/2018/11/transcoding-grpc-to-http-json-using-envoy/)
