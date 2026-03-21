- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
- [HTTP Versions](#http-versions)
  - [HTTP/1.1](#http11)
  - [HTTP/2](#http2)
    - [HTTP/2 Flow Control](#http2-flow-control)
  - [HTTP/3](#http3)
- [HTTP Flow](#http-flow)
- [HTTP 1.1 Methods](#http-11-methods)
  - [Examples](#examples)

----

# Abstract

**HTTP** stands for **HyperText Transfer Protocol**. It is a set of rules for
transferring files, such as text, images, videos, and other multimedia elements,
on the World Wide Web. When a user opens a web browser and types a URL (Uniform
Resource Locator), the browser sends an HTTP request to the web server to fetch
the information and display it on the user's screen. HTTP is an application
layer protocol built on top of the TCP/IP suite, allowing clients and servers to
communicate seamlessly across the internet.

# Materials

* [HTTP/2 알아보기 - 1편 @ whatap](https://www.whatap.io/ko/blog/38/)
* [[초보개발자 일지] HTTP 프로토콜의 이해 — 1 (HTTP 정의, HTTP/1.1)](https://medium.com/@shaul1991/%EC%B4%88%EB%B3%B4%EA%B0%9C%EB%B0%9C%EC%9E%90-%EC%9D%BC%EC%A7%80-http-%ED%94%84%EB%A1%9C%ED%86%A0%EC%BD%9C%EC%9D%98-%EC%9D%B4%ED%95%B4-1-b9005a77e5fd)
* [Hypertext Transfer Protocol @ wikipedia](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol)
* [HTTP1.1 vs HTTP2.0 차이점 간단히 살펴보기](https://medium.com/@shlee1353/http1-1-vs-http2-0-%EC%B0%A8%EC%9D%B4%EC%A0%90-%EA%B0%84%EB%8B%A8%ED%9E%88-%EC%82%B4%ED%8E%B4%EB%B3%B4%EA%B8%B0-5727b7499b78)
* [SPDY는 무엇인가?](https://d2.naver.com/helloworld/140351)
* [Hypertext Transfer Protocol Version 2 (HTTP/2) @ RFC](https://tools.ietf.org/html/rfc7540)
* [What is HTTP/2 – The Ultimate Guide](https://kinsta.com/learn/what-is-http2/)
* [HTTP 응답코드 메소드 정리 GET, POST, PUT, PATCH, DELETE, TRACE, OPTIONS](https://javaplant.tistory.com/18)

# Basic

History of HTTP 

| Year |	HTTP Version |
|--|--|
| 1991	| 0.9 |
| 1996	| 1.0 |
| 1997	| 1.1 |
| 2015	| 2.0 |
| Draft (2020)	| 3.0 |

# HTTP Versions

## HTTP/1.1

HTTP/1.1 is the first major revision of the original HTTP (HTTP/1.0). It was
introduced in 1997 and aimed to improve the efficiency and reliability of data
transmission over the internet. Some key features of HTTP/1.1 include:

- **Persistent connections**: Unlike HTTP/1.0, which closed connections after
  serving each request, HTTP/1.1 introduced persistent connections, allowing
  multiple requests to be sent over a single connection, reducing the overhead
  of establishing and closing connections.
- **Pipelining**: This feature allows clients to send multiple requests without
  waiting for each response, which reduces the latency associated with multiple
  round-trips between the client and server.
- **Chunked transfer encoding**: This allows servers to send responses in
  smaller chunks, so clients can start processing data before the entire
  response is received.
- **Additional caching mechanisms**: HTTP/1.1 introduced more advanced caching
  features, such as the ETag header and Cache-Control directives, to help
  minimize network traffic and improve performance.
- **Improved request and response headers**: HTTP/1.1 added several new headers
  and standardized the formatting and interpretation of existing ones.

## HTTP/2

HTTP/2 is the second major version of HTTP, published in 2015, and was designed
to address the limitations and performance issues of HTTP/1.1. Key features of
HTTP/2 include:

- **Multiplexing**: In HTTP/2, multiple requests and responses can be sent
  concurrently over a single TCP connection. This reduces the overhead and
  latency associated with managing multiple connections.
- **Server push**: This allows the server to proactively send resources to the
  client's cache before the client requests them, which can improve page load
  times.
- **Header compression**: HTTP/2 introduces the HPACK compression algorithm,
  which reduces the size of request and response headers, resulting in less
  bandwidth usage and faster transmissions.
- **Binary framing**: HTTP/2 uses a binary format for data transmission, making
  it more efficient and less error-prone compared to the text-based format used
  in HTTP/1.1.
- **Stream prioritization**: Clients can indicate the priority of specific
  resources when making requests, allowing servers to send the most important
  data first.

### HTTP/2 Flow Control

HTTP/2는 하나의 TCP 연결 위에 여러 스트림을 멀티플렉싱한다. 이때 한 스트림이
대역폭을 독점하지 못하도록 **애플리케이션 레벨의 flow control**이 존재한다. 이것은
TCP flow control (L4, OS 커널 관리)과는 별개의 메커니즘이다.

```
┌─────────────────────────────────┐
│  Application (gRPC, HTTP API)   │
├─────────────────────────────────┤
│  HTTP/2 Flow Control            │  L7, 초기 window 64KB (RFC 7540)
├─────────────────────────────────┤
│  TLS                            │
├─────────────────────────────────┤
│  TCP Flow Control               │  L4, OS 커널 관리 (rwnd)
├─────────────────────────────────┤
│  IP                             │
└─────────────────────────────────┘
```

**두 가지 Window:**

- **Stream-level window**: 개별 스트림(요청/응답)당 초기 64KB
- **Connection-level window**: 연결 전체에 대해 초기 64KB

수신자가 DATA 프레임을 받으면 window 크기가 줄어들고, 데이터를 처리한 후
`WINDOW_UPDATE` 프레임을 보내야 window가 복구된다.

```
Sender                            Receiver
    │                                 │
    │── DATA (16KB) ──────────────▶  │  window: 64KB → 48KB
    │── DATA (16KB) ──────────────▶  │  window: 48KB → 32KB
    │── DATA (16KB) ──────────────▶  │  window: 32KB → 16KB
    │── DATA (16KB) ──────────────▶  │  window: 16KB → 0KB
    │                                 │
    │   (window 소진, 전송 중단)       │
    │                                 │
    │◀── WINDOW_UPDATE (64KB) ──── │  수신자가 처리 완료 후 window 복구
    │                                 │  window: 0KB → 64KB
    │── DATA (계속 전송) ─────────▶  │
```

**Flow Control 상태 (Envoy 기준):**

| 상태 | 설명 |
|------|------|
| **Backed Up** | Connection-level window가 소진되어 upstream 전송 완전 차단. Downstream 읽기도 일시 중지 (backpressure 전파) |
| **Paused Reading** | Window 소진으로 downstream 읽기 일시 중지 |
| **Resumed Reading** | Window 복구 후 읽기 재개 |

**주요 발생 원인:**

| 원인 | 설명 |
|------|------|
| 수신자 처리 지연 | 목적지 서비스가 데이터를 빠르게 소비하지 못함 (CPU 부족, GC 등) |
| 초기 window 크기 제약 | 64KB는 고대역폭 환경에서 금방 소진됨 |
| WINDOW_UPDATE 지연 | 수신자가 WINDOW_UPDATE를 늦게 보냄 |
| 멀티플렉싱 과다 | 한 연결에 너무 많은 스트림이 몰림 |

**TCP Flow Control 과의 차이:**

| 구분 | TCP Flow Control | HTTP/2 Flow Control |
|------|-----------------|-------------------|
| 계층 | L4 (Transport) | L7 (Application) |
| 관리 주체 | OS 커널 | 애플리케이션 (Envoy, nginx 등) |
| 단위 | 바이트 스트림 | 스트림 / 연결 |
| Window 크기 | OS가 동적 조절 (수십 KB ~ 수 MB) | 초기 64KB (RFC 7540 기본값, `SETTINGS_INITIAL_WINDOW_SIZE`로 조절 가능) |
| 프레임 | TCP ACK + Window Size | `WINDOW_UPDATE` 프레임 |

**Envoy Flow Control 메트릭 (Service Mesh 모니터링):**

| 메트릭 | 설명 |
|--------|------|
| `envoy_cluster_upstream_flow_control_backed_up_total` | Connection-level window 소진으로 완전 차단된 횟수 |
| `envoy_cluster_upstream_flow_control_paused_reading_total` | Window 소진으로 읽기 일시 중지된 횟수 |
| `envoy_cluster_upstream_flow_control_resumed_reading_total` | Window 복구 후 읽기 재개된 횟수 |

모니터링 시 `rate(...[5m])`으로 초당 발생 빈도(ops/s)를 확인한다. `sum()`은
모든 pod 합계이므로, pod별 상황은 `by (pod)` 또는 `avg()`로 분리해서 봐야 한다.

**패널 간 정상 관계:**

| 관계 | 설명 |
|------|------|
| Paused ≈ Resumed | Pause가 발생하면 반드시 Resume이 뒤따르므로 rate가 거의 일치해야 정상 |
| Backed Up ≤ Paused | Backed Up은 connection-level 소진이라 더 심각한 상태. Paused보다 같거나 낮아야 정상 |
| Backed Up ≈ Paused | 거의 매번 connection-level까지 막히는 상태. 심각한 backpressure |
| 모두 0 | 정상. Flow control 이슈 없음 |

**Backed Up 발생 시 트러블슈팅:**

1단계 - 어디서 발생하는지 좁히기:

```promql
-- pod별 확인: 특정 pod에 집중되는지
sort_desc(rate(envoy_cluster_upstream_flow_control_backed_up_total{namespace="$source"}[5m]))

-- upstream cluster별 확인: 어떤 destination 서비스로 가는 트래픽인지
sum by (envoy_cluster_name)(rate(envoy_cluster_upstream_flow_control_backed_up_total{namespace="$source"}[5m]))
```

2단계 - 근본 원인 파악:

| 원인 | 확인 방법 | 해결 |
|------|----------|------|
| 수신자가 느림 | upstream 서비스의 CPU/메모리/GC 확인, latency P99 급등 여부 | 서비스 스케일업/아웃, GC 튜닝, 코드 최적화 |
| HTTP/2 window가 너무 작음 | 대용량 payload를 주고받는 서비스인지 확인 | Envoy window 크기 증가 |
| 한 연결에 스트림 과다 | `envoy_cluster_upstream_cx_active`, `envoy_cluster_upstream_rq_active` 확인 | `max_concurrent_streams` 제한, 연결 수 늘리기 |
| 네트워크 지연 | pod 간 RTT 확인, 다른 AZ로 가는 트래픽인지 확인 | locality-aware routing 적용 |

3단계 - Envoy 설정 튜닝 (HTTP/2 window 크기 증가):

```yaml
# Envoy cluster 설정
clusters:
- name: upstream_service
  http2_protocol_options:
    initial_stream_window_size: 1048576     # 1MB (기본 64KB)
    initial_connection_window_size: 1048576  # 1MB (기본 64KB)
```

Istio 환경이라면 DestinationRule로 설정:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: upstream-service
spec:
  host: upstream-service.namespace.svc.cluster.local
  trafficPolicy:
    connectionPool:
      http:
        h2UpgradePolicy: UPGRADE
        maxRequestsPerConnection: 1000  # 연결당 요청 수 제한
      tcp:
        maxConnections: 100             # 연결 수 늘리기
```

함께 확인할 메트릭:

| 메트릭 | 목적 |
|--------|------|
| `envoy_cluster_upstream_cx_active` | 활성 연결 수 |
| `envoy_cluster_upstream_rq_active` | 활성 요청 수 |
| `envoy_cluster_upstream_rq_time` | upstream 응답 시간 |
| `envoy_cluster_upstream_cx_rx_bytes_total` | 수신 바이트 (대용량 payload 여부) |
| `container_cpu_usage_seconds_total` | upstream pod CPU 사용량 |
| `container_memory_working_set_bytes` | upstream pod 메모리 사용량 |

트러블슈팅 우선순위:

```
1. 수신자(upstream) 서비스가 느린가?  → 서비스 자체 최적화/스케일링
2. payload가 큰가?                    → window 크기 증가
3. 연결당 스트림이 너무 많은가?        → 연결 수 늘리기
4. 네트워크 문제인가?                  → locality routing
```

대부분의 경우 1번(upstream 서비스 성능)이 근본 원인이고, 2번(window 크기 증가)이
가장 빠른 임시 완화책이다.

**참고:**
- [RFC 7540 Section 5.2 - Flow Control](https://tools.ietf.org/html/rfc7540#section-5.2)
- [Envoy HTTP/2 Flow Control Documentation](https://www.envoyproxy.io/docs/envoy/latest/configuration/best_practices/edge)

## HTTP/3

HTTP/3 is the latest major version of HTTP, which is currently in the
implementation stage. It builds on top of HTTP/2 but replaces the underlying
transport protocol, TCP, with QUIC (Quick UDP Internet Connections). QUIC was
primarily developed by Google to improve the performance of web applications and
address the issues of network latency, head-of-line blocking, and connection
reliability. Main features of HTTP/3 include:

- **QUIC protocol**: HTTP/3 uses QUIC, a transport protocol that operates over
  UDP instead of TCP, which provides faster connection establishment and
  improved resilience against packet loss and network congestion.
- **Improved multiplexing**: With QUIC, data streams are independent, so even if
  a packet is lost or delayed in one stream, it does not affect the transmission
  of data in other streams.
- **Connection migration**: QUIC can seamlessly transition between different IP
  addresses or network interfaces, offering better performance on mobile devices
  and reducing the impact of network changes on user experience.
- **Enhanced security**: QUIC includes built-in transport layer encryption,
  making it more secure by default. This also eliminates the need for an
  additional TLS (Transport Layer Security) handshake, decreasing connection
  setup time.

HTTP/3 maintains the features introduced in HTTP/2, such as server push, header
compression, and binary framing, but improves upon them with the benefits
provided by the QUIC protocol.

# HTTP Flow

* [What happens when... | github](https://github.com/alex/what-happens-when)
  * [...하면 생기는 일 | github](https://github.com/SantonyChoi/what-happens-when-KR)

----

**The "g" key is pressed**

**The "enter" key bottoms out**

**Interrupt fires [NOT for USB keyboards]**

* "g" 키를 누르면 인터럽트가 발생합니다. 그리고 kernel 이 인터럽트 핸들러를 호출합니다.

**(On Windows) A WM_KEYDOWN message is sent to the app**

* kernel 이 WM_KEYDOWN 메시지를 Browser application 으로 전달합니다.

**Parse URL**

**Is it a URL or a search term?**

* Browser application 은 URL 을 parsing 합니다. 그리고 protocol 혹은 valid domain name 이 아니면 default search engine 에게 HTTP Request 를 전송합니다.

**Convert non-ASCII Unicode characters in hostname**

* Browser application 은 URL 의 host name 에 `a-z, A-Z, 0-9, -, .` 아닌 문자열이 있는지 확인합니다. Unicode 가 있을 때는 [Punycode encoding](https://en.wikipedia.org/wiki/Punycode) 을 하기도 한다.

**Check HSTS list**

* Browser application 는 `HSTS (HTTP Strict Transport Security)` 에 URL 이 있는지 검사합니다. 있다면 HTTP 대신 HTTPS Request 를 해야합니다.

**DNS lookup**

* Browser application 은 Domain Cache 에 host name 이 있는지 검사합니다.
  * Chrome 의 경우 `chrome://net-internals/#dns` 에서 DNS cache 를 확인할 수 있다.
* DNS cache miss 가 발생하면 `gethostbyname` 을 호출합니다.
* `gethostbyname` 은 `/etc/hosts` 를 검색합니다.
* `/etc/hosts` 에 없다면 `gethostbyname` 은 DNS query 를 합니다.

**ARP process**

**Opening of a socket**

**TLS handshake**

**If a packet is dropped**

**HTTP protocol**

**HTTP Server Request Handle**

**Behind the scenes of the Browser**

**Browser**

**HTML parsing**

**CSS interpretation**

**Page Rendering**

**GPU Rendering**

**Post-rendering and user-induced execution**

# HTTP 1.1 Methods

- HTTP GET: This method retrieves a resource from the server. It is idempotent,
  which means that making multiple identical GET requests will yield the same
  result each time. GET is often used to request web pages, images, and other
  static files from a server.
- HTTP PUT: PUT updates or creates a resource on the server. It is idempotent,
  so multiple identical PUT requests will have the same effect as a single
  request. This method is often used for updating data on a server, where
  providing the complete updated record ensures consistency.
- HTTP POST: POST is used to create new resources on the server. Unlike GET and
  PUT, it is not idempotent, meaning that making two identical POST requests
  will result in two duplicates of the resource being created. POST is typically
  used for submitting forms or creating new entries in a database.
- HTTP DELETE: DELETE is used to remove a resource from the server. Like GET and
  PUT, it is idempotent – multiple identical DELETE requests will only remove
  the resource once. This is frequently used to delete files, database entries,
  or other data objects on a server.
- HTTP PATCH: PATCH applies partial modifications to a resource on the server,
  as opposed to PUT, which updates the entire resource. This method is useful
  when you want to update only specific attributes of a resource without
  affecting the other attributes.
- HTTP HEAD: HEAD requests a response identical to a GET request, but without
  including the response body. This method is useful for checking if a resource
  exists or retrieving metadata, such as content-length or content-type, without
  downloading the entire resource.
- HTTP CONNECT: CONNECT establishes a network connection to the server
  identified by the target resource, typically for use with network protocols
  like SSL or TLS. This method is often used by web proxies to enable secure
  connections between clients and servers.
- HTTP OPTIONS: OPTIONS describes the communication options available for the
  target resource, such as the HTTP methods supported or any custom headers that
  may be required. This can be used by clients to discover information about a
  server's capabilities or configuration.
- HTTP TRACE: TRACE performs a message loop-back test along the path to the
  target resource, returning the request/response message as the response body.
  This can be useful for testing or debugging purposes to see the series of
  headers and intermediaries a request passes through on its way to the server.

## Examples

```c
// Example 1: HTTP GET
Request:
GET /index.html HTTP/1.1
Host: www.example.com

Response:
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html> <html> <head> <title>Example Page</title> </head> <body> <h1>Welcome to www.example.com</h1> </body> </html> ```

// Example 2: HTTP PUT
Request:

PUT /api/users/123 HTTP/1.1
Host: www.example.com
Content-Type: application/json

{
    "name": "John Doe",
    "email": "john@example.com"
}

Response:

HTTP/1.1 200 OK
Content-Type: application/json

{
    "message": "User successfully updated."
}

//Example 3: HTTP POST
Request:

POST /api/users HTTP/1.1
Host: www.example.com
Content-Type: application/json

{
    "name": "Jane Doe",
    "email": "jane@example.com"
}
Response:

HTTP/1.1 201 Created
Content-Type: application/json

{
    "id": 456,
    "name": "Jane Doe",
    "email": "jane@example.com"
}

//Example 4: HTTP DELETE
Request:

DELETE /api/users/123 HTTP/1.1
Host: www.example.com
Response:

HTTP/1.1 200 OK
Content-Type: application/json

{
    "message": "User successfully deleted."
}

//Example 5: HTTP HEAD
Request:

HEAD /index.html HTTP/1.1
Host: www.example.com

Response:

HTTP/1.1 200 OK
Content-Type: text/html

//Example 6: HTTP PATCH
Request:

PATCH /api/users/123 HTTP/1.1
Host: www.example.com
Content-Type: application/json

{
    "email": "new-email@example.com"
}

Response:

HTTP/1.1 200 OK
Content-Type: application/json

{
    "message": "User email successfully updated."
}

// CONNECT example:
Request:

CONNECT www.example.com:443 HTTP/1.1
Host: www.example.com
Proxy-Authorization: Basic abc123xyz
Response:

HTTP/1.1 200 Connection Established
Proxy-agent: ProxyServer/1.0

// OPTIONS example:
Request:

OPTIONS /my-resource-path HTTP/1.1
Host: www.example.com

Response:

HTTP/1.1 200 OK
Allow: GET, POST, PUT, DELETE, OPTIONS
Content-Length: 0

// TRACE example:
Request:

TRACE /my-resource-path HTTP/1.1
Host: www.example.com

Response:

HTTP/1.1 200 OK
Content-Type: message/http
Content-Length: [length of the response body]

TRACE /my-resource-path HTTP/1.1
Host: www.example.com
```