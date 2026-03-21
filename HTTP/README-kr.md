- [개요](#개요)
- [자료](#자료)
- [기본](#기본)
- [HTTP 버전](#http-버전)
  - [HTTP/1.1](#http11)
  - [HTTP/2](#http2)
    - [HTTP/2 Flow Control](#http2-flow-control)
  - [HTTP/3](#http3)
- [HTTP 흐름](#http-흐름)
- [HTTP 1.1 메서드](#http-11-메서드)
  - [예제](#예제)

----

# 개요

**HTTP**는 **HyperText Transfer Protocol**의 약자이다. 텍스트, 이미지, 동영상 등
멀티미디어 요소를 월드 와이드 웹(WWW)에서 전송하기 위한 규칙의 집합이다. 사용자가
웹 브라우저를 열고 URL(Uniform Resource Locator)을 입력하면, 브라우저는 웹 서버에
HTTP 요청을 보내 정보를 가져와 화면에 표시한다. HTTP는 TCP/IP 위에 구축된 애플리케이션
계층 프로토콜로, 클라이언트와 서버가 인터넷을 통해 원활하게 통신할 수 있게 해준다.

# 자료

* [HTTP/2 알아보기 - 1편 @ whatap](https://www.whatap.io/ko/blog/38/)
* [[초보개발자 일지] HTTP 프로토콜의 이해 — 1 (HTTP 정의, HTTP/1.1)](https://medium.com/@shaul1991/%EC%B4%88%EB%B3%B4%EA%B0%9C%EB%B0%9C%EC%9E%90-%EC%9D%BC%EC%A7%80-http-%ED%94%84%EB%A1%9C%ED%86%A0%EC%BD%9C%EC%9D%98-%EC%9D%B4%ED%95%B4-1-b9005a77e5fd)
* [Hypertext Transfer Protocol @ wikipedia](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol)
* [HTTP1.1 vs HTTP2.0 차이점 간단히 살펴보기](https://medium.com/@shlee1353/http1-1-vs-http2-0-%EC%B0%A8%EC%9D%B4%EC%A0%90-%EA%B0%84%EB%8B%A8%ED%9E%88-%EC%82%B4%ED%8E%B4%EB%B3%B4%EA%B8%B0-5727b7499b78)
* [SPDY는 무엇인가?](https://d2.naver.com/helloworld/140351)
* [Hypertext Transfer Protocol Version 2 (HTTP/2) @ RFC](https://tools.ietf.org/html/rfc7540)
* [What is HTTP/2 – The Ultimate Guide](https://kinsta.com/learn/what-is-http2/)
* [HTTP 응답코드 메소드 정리 GET, POST, PUT, PATCH, DELETE, TRACE, OPTIONS](https://javaplant.tistory.com/18)

# 기본

HTTP 역사

| 연도 | HTTP 버전 |
|--|--|
| 1991 | 0.9 |
| 1996 | 1.0 |
| 1997 | 1.1 |
| 2015 | 2.0 |
| Draft (2020) | 3.0 |

# HTTP 버전

## HTTP/1.1

HTTP/1.1은 최초의 HTTP(HTTP/1.0)에 대한 첫 번째 주요 개정판이다. 1997년에
도입되었으며, 인터넷을 통한 데이터 전송의 효율성과 신뢰성을 개선하는 것을
목표로 했다. HTTP/1.1의 주요 특징은 다음과 같다:

- **지속 연결 (Persistent connections)**: HTTP/1.0은 각 요청을 처리한 후 연결을
  닫았지만, HTTP/1.1은 지속 연결을 도입하여 하나의 연결로 여러 요청을 보낼 수
  있게 했다. 이를 통해 연결 설정/해제 오버헤드가 줄어든다.
- **파이프라이닝 (Pipelining)**: 클라이언트가 각 응답을 기다리지 않고 여러 요청을
  연속으로 보낼 수 있어, 다중 왕복에 따른 지연을 줄여준다.
- **청크 전송 인코딩 (Chunked transfer encoding)**: 서버가 응답을 작은 청크로
  나누어 보낼 수 있어, 전체 응답을 받기 전에 클라이언트가 데이터 처리를 시작할
  수 있다.
- **향상된 캐싱 메커니즘**: ETag 헤더, Cache-Control 디렉티브 등 더 발전된 캐싱
  기능을 도입하여 네트워크 트래픽을 최소화하고 성능을 개선했다.
- **개선된 요청/응답 헤더**: 여러 새로운 헤더를 추가하고, 기존 헤더의 형식과
  해석을 표준화했다.

## HTTP/2

HTTP/2는 2015년에 발표된 HTTP의 두 번째 주요 버전으로, HTTP/1.1의 한계와 성능
문제를 해결하기 위해 설계되었다. HTTP/2의 주요 특징은 다음과 같다:

- **멀티플렉싱 (Multiplexing)**: 하나의 TCP 연결 위에서 여러 요청과 응답을 동시에
  전송할 수 있다. 이를 통해 다중 연결 관리에 따른 오버헤드와 지연이 줄어든다.
- **서버 푸시 (Server push)**: 서버가 클라이언트의 요청 전에 미리 리소스를
  클라이언트 캐시로 전송할 수 있어, 페이지 로드 시간을 개선할 수 있다.
- **헤더 압축 (Header compression)**: HPACK 압축 알고리즘을 도입하여 요청/응답
  헤더 크기를 줄이고, 대역폭 사용량을 낮추며 전송 속도를 향상시켰다.
- **바이너리 프레이밍 (Binary framing)**: HTTP/1.1의 텍스트 기반 형식 대신
  바이너리 형식을 사용하여 데이터 전송을 더 효율적이고 오류에 강하게 했다.
- **스트림 우선순위 (Stream prioritization)**: 클라이언트가 요청 시 특정 리소스의
  우선순위를 지정할 수 있어, 서버가 가장 중요한 데이터를 먼저 보낼 수 있다.

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

HTTP/3는 현재 구현 단계에 있는 HTTP의 최신 주요 버전이다. HTTP/2를 기반으로
하되, 기존의 전송 프로토콜인 TCP를 QUIC(Quick UDP Internet Connections)으로
대체한다. QUIC는 주로 Google에서 개발한 것으로, 웹 애플리케이션의 성능을
개선하고 네트워크 지연, Head-of-Line 블로킹, 연결 안정성 문제를 해결하는 것을
목표로 한다. HTTP/3의 주요 특징은 다음과 같다:

- **QUIC 프로토콜**: TCP 대신 UDP 위에서 동작하는 전송 프로토콜인 QUIC를
  사용하여, 더 빠른 연결 설정과 패킷 손실 및 네트워크 혼잡에 대한 향상된
  복원력을 제공한다.
- **개선된 멀티플렉싱**: QUIC에서는 데이터 스트림이 독립적이므로, 한 스트림에서
  패킷이 손실되거나 지연되더라도 다른 스트림의 데이터 전송에 영향을 주지 않는다.
- **연결 마이그레이션 (Connection migration)**: QUIC는 서로 다른 IP 주소나
  네트워크 인터페이스 간에 원활하게 전환할 수 있어, 모바일 기기에서 더 나은
  성능을 제공하고 네트워크 변경이 사용자 경험에 미치는 영향을 줄인다.
- **강화된 보안**: QUIC에는 전송 계층 암호화가 내장되어 있어 기본적으로 더
  안전하다. 이를 통해 별도의 TLS(Transport Layer Security) 핸드셰이크가
  불필요해지며 연결 설정 시간이 단축된다.

HTTP/3는 서버 푸시, 헤더 압축, 바이너리 프레이밍 등 HTTP/2에서 도입된 기능을
유지하면서도 QUIC 프로토콜이 제공하는 이점으로 이를 개선한다.

# HTTP 흐름

* [What happens when... | github](https://github.com/alex/what-happens-when)
  * [...하면 생기는 일 | github](https://github.com/SantonyChoi/what-happens-when-KR)

----

**"g" 키를 누른다**

**"enter" 키를 끝까지 누른다**

**인터럽트 발생 [USB 키보드 제외]**

* "g" 키를 누르면 인터럽트가 발생합니다. 그리고 kernel 이 인터럽트 핸들러를 호출합니다.

**(Windows) WM_KEYDOWN 메시지가 앱으로 전달된다**

* kernel 이 WM_KEYDOWN 메시지를 Browser application 으로 전달합니다.

**URL 파싱**

**URL인가 검색어인가?**

* Browser application 은 URL 을 parsing 합니다. 그리고 protocol 혹은 valid domain name 이 아니면 default search engine 에게 HTTP Request 를 전송합니다.

**호스트명의 비ASCII 유니코드 문자 변환**

* Browser application 은 URL 의 host name 에 `a-z, A-Z, 0-9, -, .` 아닌 문자열이 있는지 확인합니다. Unicode 가 있을 때는 [Punycode encoding](https://en.wikipedia.org/wiki/Punycode) 을 하기도 한다.

**HSTS 목록 확인**

* Browser application 는 `HSTS (HTTP Strict Transport Security)` 에 URL 이 있는지 검사합니다. 있다면 HTTP 대신 HTTPS Request 를 해야합니다.

**DNS 조회**

* Browser application 은 Domain Cache 에 host name 이 있는지 검사합니다.
  * Chrome 의 경우 `chrome://net-internals/#dns` 에서 DNS cache 를 확인할 수 있다.
* DNS cache miss 가 발생하면 `gethostbyname` 을 호출합니다.
* `gethostbyname` 은 `/etc/hosts` 를 검색합니다.
* `/etc/hosts` 에 없다면 `gethostbyname` 은 DNS query 를 합니다.

**ARP 프로세스**

**소켓 열기**

**TLS 핸드셰이크**

**패킷이 손실된 경우**

**HTTP 프로토콜**

**HTTP 서버 요청 처리**

**브라우저 내부 동작**

**브라우저**

**HTML 파싱**

**CSS 해석**

**페이지 렌더링**

**GPU 렌더링**

**렌더링 후 처리 및 사용자 유발 실행**

# HTTP 1.1 메서드

- HTTP GET: 서버에서 리소스를 조회한다. 멱등(idempotent)하므로, 동일한 GET
  요청을 여러 번 보내도 매번 같은 결과를 얻는다. 웹 페이지, 이미지 등 정적
  파일을 요청할 때 주로 사용한다.
- HTTP PUT: 서버의 리소스를 갱신하거나 생성한다. 멱등하므로, 동일한 PUT 요청을
  여러 번 보내도 한 번 보낸 것과 같은 효과를 가진다. 완전한 갱신 레코드를
  제공하여 일관성을 보장하는 데이터 갱신에 주로 사용한다.
- HTTP POST: 서버에 새로운 리소스를 생성한다. GET, PUT과 달리 멱등하지 않으므로,
  동일한 POST 요청을 두 번 보내면 리소스가 두 개 생성된다. 폼 제출이나 데이터베이스에
  새 항목을 만들 때 주로 사용한다.
- HTTP DELETE: 서버의 리소스를 삭제한다. GET, PUT과 마찬가지로 멱등하여, 동일한
  DELETE 요청을 여러 번 보내도 리소스는 한 번만 삭제된다. 파일, 데이터베이스
  항목 등을 삭제할 때 사용한다.
- HTTP PATCH: 서버의 리소스에 부분적인 수정을 적용한다. 전체 리소스를 갱신하는
  PUT과 달리, 특정 속성만 변경하고 싶을 때 유용하다.
- HTTP HEAD: GET 요청과 동일한 응답을 요청하되, 응답 본문(body)은 포함하지
  않는다. 리소스 존재 여부 확인이나 content-length, content-type 등 메타데이터를
  전체 리소스를 다운로드하지 않고 조회할 때 유용하다.
- HTTP CONNECT: 대상 리소스가 식별하는 서버로의 네트워크 연결을 설정한다. 주로
  SSL이나 TLS 같은 네트워크 프로토콜에 사용되며, 웹 프록시가 클라이언트와 서버
  간의 보안 연결을 가능하게 할 때 사용한다.
- HTTP OPTIONS: 대상 리소스에 대해 사용 가능한 통신 옵션을 설명한다. 지원하는
  HTTP 메서드나 필요한 커스텀 헤더 등을 포함한다. 서버의 기능이나 설정에 대한
  정보를 확인하는 데 사용한다.
- HTTP TRACE: 대상 리소스까지의 경로를 따라 메시지 루프백 테스트를 수행하며,
  요청/응답 메시지를 응답 본문으로 반환한다. 테스트나 디버깅 목적으로, 요청이
  서버에 도달하기까지 거치는 헤더와 중간 장치들을 확인할 때 유용하다.

## 예제

```
// 예제 1: HTTP GET
요청:
GET /index.html HTTP/1.1
Host: www.example.com

응답:
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html> <html> <head> <title>Example Page</title> </head> <body> <h1>Welcome to www.example.com</h1> </body> </html>

// 예제 2: HTTP PUT
요청:

PUT /api/users/123 HTTP/1.1
Host: www.example.com
Content-Type: application/json

{
    "name": "John Doe",
    "email": "john@example.com"
}

응답:

HTTP/1.1 200 OK
Content-Type: application/json

{
    "message": "User successfully updated."
}

// 예제 3: HTTP POST
요청:

POST /api/users HTTP/1.1
Host: www.example.com
Content-Type: application/json

{
    "name": "Jane Doe",
    "email": "jane@example.com"
}

응답:

HTTP/1.1 201 Created
Content-Type: application/json

{
    "id": 456,
    "name": "Jane Doe",
    "email": "jane@example.com"
}

// 예제 4: HTTP DELETE
요청:

DELETE /api/users/123 HTTP/1.1
Host: www.example.com

응답:

HTTP/1.1 200 OK
Content-Type: application/json

{
    "message": "User successfully deleted."
}

// 예제 5: HTTP HEAD
요청:

HEAD /index.html HTTP/1.1
Host: www.example.com

응답:

HTTP/1.1 200 OK
Content-Type: text/html

// 예제 6: HTTP PATCH
요청:

PATCH /api/users/123 HTTP/1.1
Host: www.example.com
Content-Type: application/json

{
    "email": "new-email@example.com"
}

응답:

HTTP/1.1 200 OK
Content-Type: application/json

{
    "message": "User email successfully updated."
}

// CONNECT 예제:
요청:

CONNECT www.example.com:443 HTTP/1.1
Host: www.example.com
Proxy-Authorization: Basic abc123xyz

응답:

HTTP/1.1 200 Connection Established
Proxy-agent: ProxyServer/1.0

// OPTIONS 예제:
요청:

OPTIONS /my-resource-path HTTP/1.1
Host: www.example.com

응답:

HTTP/1.1 200 OK
Allow: GET, POST, PUT, DELETE, OPTIONS
Content-Length: 0

// TRACE 예제:
요청:

TRACE /my-resource-path HTTP/1.1
Host: www.example.com

응답:

HTTP/1.1 200 OK
Content-Type: message/http
Content-Length: [응답 본문의 길이]

TRACE /my-resource-path HTTP/1.1
Host: www.example.com
```