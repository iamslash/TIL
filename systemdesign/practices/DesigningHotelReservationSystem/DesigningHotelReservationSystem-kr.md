# 목차

- [요구사항](#요구사항)
  - [기능적 요구사항](#기능적-요구사항)
  - [비기능적 요구사항](#비기능적-요구사항)
  - [용량 산정](#용량-산정)
- [High Level Design](#high-level-design)
  - [API 설계](#api-설계)
  - [데이터 모델 설계](#데이터-모델-설계)
  - [High-Level 아키텍처](#high-level-아키텍처)
- [Design Deep Dive](#design-deep-dive)
  - [호텔 검색 및 발견](#호텔-검색-및-발견)
  - [객실 타입별 예약](#객실-타입별-예약)
  - [Room Assignment Flow](#room-assignment-flow)
  - [동시성 문제](#동시성-문제)
  - [결제 통합](#결제-통합)
  - [확장성](#확장성)
    - [Database Sharding](#database-sharding)
    - [Caching](#caching)
  - [서비스 간 데이터 일관성](#서비스-간-데이터-일관성)
  - [예약 상태 관리](#예약-상태-관리)
  - [모니터링 및 관찰성](#모니터링-및-관찰성)
  - [보안 고려사항](#보안-고려사항)
  - [장애 처리 및 복구](#장애-처리-및-복구)
  - [Edge Cases](#edge-cases)
- [완전한 예약 플로우](#완전한-예약-플로우)
- [인터뷰 질문](#인터뷰-질문)
- [References](#references)

----

# 요구사항

## 기능적 요구사항

* 호텔 관련 페이지를 표시한다.
* 호텔 객실 상세 페이지를 표시한다.
* 객실을 예약한다.
* 관리자 콘솔에서 호텔 또는 객실 정보를 추가/삭제/수정한다.
* 오버부킹을 지원한다 (재고의 110%까지 예약 허용).
* 호텔 객실 가격은 날짜별로 변경될 수 있다.
* 사용자는 호텔을 검색하고 필터링할 수 있다 (위치, 날짜, 객실 타입 등).
* 예약을 취소하거나 수정할 수 있다.
* 예약 확인 및 알림을 발송한다.

## 비기능적 요구사항

* **높은 동시성**
  * 많은 사용자가 동시에 같은 객실을 예약하려고 시도할 수 있다.
  * Race condition을 적절히 처리해야 한다.
* **적절한 지연시간**
  * 예약 프로세스는 빠르고 반응적이어야 한다.
  * p99 레이턴시는 500ms 이내를 목표로 한다.
* **높은 가용성**
  * 시스템은 99.9% 이상의 가동시간을 유지해야 한다.
* **데이터 일관성**
  * 이중 예약이 발생하지 않아야 한다.
  * 재고 정보는 정확해야 한다.
* **확장성**
  * Booking.com처럼 트래픽이 급증해도 대응할 수 있어야 한다.

## 용량 산정

| 수치 | 설명 | 계산 |
|--|---|--|
| 5,000 | 호텔 수 | |
| 1,000,000 | 전체 객실 수 | |
| 70% | 평균 점유율 | |
| 3일 | 평균 숙박 기간 | |
| 233,333 | 일일 예약 건수 | 1,000,000 × 0.7 / 3 ≈ 233,333 |
| 3 QPS | 예약 요청 | 233,333 / 86,400초 ≈ 3 |
| 300 QPS | 호텔/객실 상세 조회 | 예약:조회 비율 1:100 가정 |
| 30 QPS | 예약 페이지 조회 | 예약:예약페이지 비율 1:10 가정 |
| 100 QPS | 검색 요청 | 조회의 1/3 가정 |

**스토리지 계산:**
- 호텔 정보: 5,000 × 10KB ≈ 50MB
- 객실 정보: 1,000,000 × 5KB ≈ 5GB
- 예약 정보 (1년): 233,333 × 365 × 1KB ≈ 85GB
- 재고 정보: 5,000 호텔 × 5 객실타입 × 365일 × 500B ≈ 4.5GB

**대역폭:**
- 쓰기: 3 QPS × 5KB ≈ 15KB/s
- 읽기: 400 QPS × 20KB ≈ 8MB/s

# High Level Design

## API 설계

### Hotel APIs

```
GET /v1/hotels/{hotel-id}
  - Description: 특정 호텔의 상세 정보 조회
  - Response: Hotel 객체 (name, address, location, amenities, images 등)

POST /v1/hotels
  - Description: 새 호텔 추가 (관리자 전용)
  - Request: Hotel 객체
  - Response: 생성된 hotel_id

PUT /v1/hotels/{hotel-id}
  - Description: 호텔 정보 수정 (관리자 전용)
  - Request: Hotel 객체 (수정할 필드만)

DELETE /v1/hotels/{hotel-id}
  - Description: 호텔 삭제 (관리자 전용)
```

### Room APIs

```
GET /v1/hotels/{hotel-id}/rooms/{room-id}
  - Description: 특정 객실 상세 정보 조회

GET /v1/hotels/{hotel-id}/room-types
  - Description: 호텔의 객실 타입 목록 조회
  - Response: 객실 타입 리스트 (타입명, 설명, amenities, 이미지 등)

POST /v1/hotels/{hotel-id}/rooms
  - Description: 새 객실 추가 (관리자 전용)

PUT /v1/hotels/{hotel-id}/rooms/{room-id}
  - Description: 객실 정보 수정 (관리자 전용)

DELETE /v1/hotels/{hotel-id}/rooms/{room-id}
  - Description: 객실 삭제 (관리자 전용)
```

### Search APIs

```
GET /v1/hotels/search
  - Description: 호텔 검색
  - Query Parameters:
    - location (required): 검색 위치 (도시, 지역 등)
    - checkInDate (required): 체크인 날짜
    - checkOutDate (required): 체크아웃 날짜
    - roomType (optional): 객실 타입
    - guests (optional): 투숙객 수
    - minPrice, maxPrice (optional): 가격 범위
    - amenities (optional): 편의시설 필터
    - sortBy (optional): 정렬 기준 (price, rating, distance)
    - page, pageSize (optional): 페이지네이션
  - Response: 검색 결과 (호텔 리스트, 가용 객실, 가격 등)

GET /v1/hotels/{hotel-id}/availability
  - Description: 특정 호텔의 객실 가용성 조회
  - Query Parameters:
    - checkInDate, checkOutDate, roomType
  - Response: 날짜별 가용 객실 수 및 가격
```

### Reservation APIs

```
GET /v1/reservations
  - Description: 사용자의 예약 내역 조회
  - Query Parameters: userId, status, page, pageSize
  - Response: 예약 리스트

GET /v1/reservations/{reservation-id}
  - Description: 특정 예약 상세 정보 조회

POST /v1/reservations
  - Description: 새 예약 생성
  - Request:
    {
        "hotelID": "333",
        "roomTypeID": "deluxe-double",
        "checkInDate": "2025-03-01",
        "checkOutDate": "2025-03-04",
        "numberOfRooms": 1,
        "guestInfo": {
            "guestID": "user123",
            "firstName": "John",
            "lastName": "Doe",
            "email": "john@example.com",
            "phone": "+1234567890"
        },
        "specialRequests": "Late check-in",
        "idempotencyKey": "uuid-12345"
    }
  - Response: reservation_id, status (pending), estimated price

PUT /v1/reservations/{reservation-id}
  - Description: 예약 수정 (날짜, 객실 타입 변경)
  - Request: 수정할 필드

DELETE /v1/reservations/{reservation-id}
  - Description: 예약 취소
```

### Payment APIs

```
POST /v1/payments
  - Description: 결제 처리
  - Request:
    {
        "reservationID": "res-12345",
        "paymentMethod": "credit_card",
        "amount": 450.00,
        "currency": "USD",
        "paymentDetails": { ... }
    }
  - Response: payment_id, status

GET /v1/payments/{payment-id}
  - Description: 결제 상태 조회
```

## 데이터 모델 설계

### Hotel Service

#### hotel 테이블
```
hotel_id (PK, varchar)
name (varchar)
address (varchar)
city (varchar, indexed)
country (varchar, indexed)
latitude (decimal)
longitude (decimal)
star_rating (tinyint)
description (text)
amenities (json)  // 예: ["WiFi", "Pool", "Gym"]
images (json)     // 이미지 URL 배열
created_at (timestamp)
updated_at (timestamp)
```

#### room_type 테이블 (새로 추가)
```
room_type_id (PK, varchar)
hotel_id (FK, varchar, indexed)
type_name (varchar)  // 예: "Deluxe Double", "Suite"
description (text)
max_occupancy (tinyint)
bed_type (varchar)  // 예: "King", "Twin"
size_sqm (smallint)
amenities (json)    // 예: ["City View", "Balcony"]
images (json)
base_price (decimal)  // 기본 가격 (실제 가격은 room_type_rate에서 관리)
created_at (timestamp)

INDEX idx_hotel_room_type (hotel_id, room_type_id)
```

#### room 테이블
```
room_id (PK, varchar)
hotel_id (FK, varchar, indexed)
room_type_id (FK, varchar, indexed)
floor (smallint)
room_number (varchar)
status (enum: 'available', 'occupied', 'maintenance', 'cleaning')
created_at (timestamp)
updated_at (timestamp)

INDEX idx_hotel_room (hotel_id, room_id)
INDEX idx_room_type (room_type_id)
```

### Rate Service

#### room_type_rate 테이블
```
id (PK, bigint, auto_increment)
hotel_id (FK, varchar)
room_type_id (FK, varchar)  // 추가됨
date (date)
rate (decimal)
currency (varchar)
created_at (timestamp)
updated_at (timestamp)

PRIMARY KEY (id)
UNIQUE KEY uk_hotel_roomtype_date (hotel_id, room_type_id, date)
INDEX idx_date_range (hotel_id, room_type_id, date)
```

### Reservation Service

#### room_type_inventory 테이블
```
hotel_id (varchar)
room_type_id (varchar)
date (date)
total_inventory (smallint)
total_reserved (smallint)
version (int, default 0)  // Optimistic locking용
created_at (timestamp)
updated_at (timestamp)

PRIMARY KEY (hotel_id, room_type_id, date)
INDEX idx_availability (hotel_id, room_type_id, date, total_inventory, total_reserved)
CHECK (total_reserved <= total_inventory * 1.1)  // 110% 오버부킹 허용
```

#### reservation 테이블
```
reservation_id (PK, varchar)
hotel_id (FK, varchar, indexed)
room_type_id (FK, varchar)
room_id (FK, varchar, nullable)  // 체크인 전에는 NULL, 이후 할당됨
guest_id (FK, varchar, indexed)
check_in_date (date)
check_out_date (date)
number_of_rooms (tinyint)
number_of_guests (tinyint)
status (enum: 'pending', 'confirmed', 'paid', 'checked_in', 'checked_out', 'canceled', 'refunded')
total_price (decimal)
currency (varchar)
special_requests (text)
created_at (timestamp)
updated_at (timestamp)
idempotency_key (varchar, unique, indexed)  // 중복 예약 방지

INDEX idx_guest_reservations (guest_id, status)
INDEX idx_hotel_date (hotel_id, check_in_date)
INDEX idx_status (status, created_at)
```

### Payment Service

#### payment 테이블
```
payment_id (PK, varchar)
reservation_id (FK, varchar, indexed)
amount (decimal)
currency (varchar)
payment_method (enum: 'credit_card', 'debit_card', 'paypal', etc.)
status (enum: 'pending', 'completed', 'failed', 'refunded')
transaction_id (varchar)  // 외부 결제 게이트웨이 트랜잭션 ID
created_at (timestamp)
updated_at (timestamp)

INDEX idx_reservation_payment (reservation_id)
INDEX idx_status (status, created_at)
```

### Guest Service

#### guest 테이블
```
guest_id (PK, varchar)
email (varchar, unique, indexed)
first_name (varchar)
last_name (varchar)
phone (varchar)
country (varchar)
date_of_birth (date)
password_hash (varchar)
created_at (timestamp)
updated_at (timestamp)

INDEX idx_email (email)
```

### 예약 상태 (status)

- **pending**: 예약 요청이 생성되었지만 결제 전
- **confirmed**: 재고 확보 완료, 결제 대기 중
- **paid**: 결제 완료
- **checked_in**: 체크인 완료
- **checked_out**: 체크아웃 완료
- **canceled**: 사용자가 취소
- **refunded**: 환불 완료

## High-Level 아키텍처

```
                                    [사용자]
                                       |
                                       v
                                   [CDN]
                         (정적 콘텐츠: 이미지, CSS, JS)
                                       |
                                       v
                              [Load Balancer]
                                       |
                                       v
                                [API Gateway]
                         (인증, Rate Limiting, 라우팅)
                                       |
        +------------------------------+--------------------------------+
        |                              |                                |
        v                              v                                v
  [Search Service]              [Hotel Service]                [Reservation Service]
  - Elasticsearch               - Hotel CRUD                    - 예약 생성/조회
  - 호텔/객실 검색              - Room CRUD                     - 재고 관리
  - 필터링/정렬                 - Room Type CRUD                - 동시성 제어
        |                              |                                |
        |                              v                                v
        |                        [Rate Service]                  [Payment Service]
        |                        - 가격 관리                      - 결제 처리
        |                        - 날짜별 요금                    - 환불 처리
        |                              |                                |
        +------------------------------+--------------------------------+
                                       |
                        +--------------+---------------+
                        |                              |
                        v                              v
                  [Message Queue]                [Cache Layer]
                  (Kafka/RabbitMQ)               (Redis Cluster)
                  - 이벤트 발행                  - 재고 캐시
                  - 비동기 처리                  - 세션 캐시
                        |                              - Rate 캐시
                        v
                [Notification Service]
                - 이메일/SMS 발송
                - 예약 확인
                        |
        +---------------+----------------+---------------+
        |               |                |               |
        v               v                v               v
   [MySQL Cluster] [MySQL Cluster]  [MySQL Cluster]  [MySQL Cluster]
   Hotel DB        Rate DB          Reservation DB   Payment DB
   (Sharded by     (Sharded by      (Sharded by      (Sharded by
    hotel_id)       hotel_id)        reservation_id)  payment_id)
        |               |                |               |
        +---------------+----------------+---------------+
                        |
                        v
                  [CDC (Debezium)]
                        |
                        v
                  [Redis Cache]
                (캐시 동기화)
```

**주요 컴포넌트 설명:**

1. **CDN (Content Delivery Network)**
   - 호텔 이미지, 프론트엔드 정적 파일 제공
   - 지역별 캐싱으로 레이턴시 감소

2. **Load Balancer**
   - L7 로드밸런서 (Application Load Balancer)
   - Health check, SSL termination
   - 트래픽 분산

3. **API Gateway**
   - 인증/인가 (JWT 토큰 검증)
   - Rate limiting (사용자당 100 req/min)
   - Request routing
   - API 버전 관리

4. **Search Service**
   - Elasticsearch 기반 전문 검색
   - 위치 기반 검색 (Geo-spatial queries)
   - 다양한 필터 및 정렬

5. **Hotel Service**
   - 호텔, 객실, 객실 타입 마스터 데이터 관리
   - CRUD 작업
   - 이미지 업로드 (S3 연동)

6. **Rate Service**
   - 날짜별 객실 타입 가격 관리
   - Dynamic pricing 지원

7. **Reservation Service**
   - 핵심 예약 로직
   - 재고 관리 (room_type_inventory)
   - 동시성 제어 (Locking)
   - Room assignment 처리

8. **Payment Service**
   - 결제 게이트웨이 연동 (Stripe, PayPal 등)
   - 결제 상태 추적
   - 환불 처리

9. **Message Queue (Kafka)**
   - 서비스 간 비동기 통신
   - 이벤트 기반 아키텍처
   - 예: ReservationCreated, PaymentCompleted 이벤트

10. **Notification Service**
    - 예약 확인 이메일/SMS
    - 체크인 리마인더
    - 결제 영수증 발송

11. **Cache Layer (Redis)**
    - 재고 정보 캐싱 (빠른 가용성 체크)
    - 객실 타입 rate 캐싱
    - 세션 정보
    - 분산 락 (Distributed Lock)

12. **CDC (Change Data Capture - Debezium)**
    - MySQL binlog를 읽어 변경사항 감지
    - Redis 캐시 자동 업데이트
    - 데이터 일관성 유지

# Design Deep Dive

## 호텔 검색 및 발견

### 검색 요구사항

사용자가 호텔을 예약하려면 먼저 검색 기능이 필요합니다:

- 위치 기반 검색 (도시, 지역, 랜드마크 근처)
- 날짜 범위 필터 (체크인/체크아웃 날짜)
- 가격 범위 필터
- 객실 타입 필터
- 편의시설 필터 (WiFi, Pool, Gym 등)
- 정렬 (가격, 평점, 거리)
- 페이지네이션

### 검색 아키텍처

**Elasticsearch 사용 이유:**
- 전문 검색 엔진으로 복잡한 쿼리 지원
- Geo-spatial 쿼리 (위도/경도 기반 검색)
- Full-text search
- Aggregation (필터 옵션 제공)
- 빠른 응답 속도 (역인덱스)

**인덱스 구조:**

```json
{
  "hotel_index": {
    "mappings": {
      "properties": {
        "hotel_id": { "type": "keyword" },
        "name": {
          "type": "text",
          "fields": {
            "keyword": { "type": "keyword" }
          }
        },
        "location": { "type": "geo_point" },
        "city": { "type": "keyword" },
        "country": { "type": "keyword" },
        "star_rating": { "type": "integer" },
        "amenities": { "type": "keyword" },
        "room_types": {
          "type": "nested",
          "properties": {
            "room_type_id": { "type": "keyword" },
            "type_name": { "type": "text" },
            "min_price": { "type": "float" }
          }
        },
        "available_dates": { "type": "date_range" }
      }
    }
  }
}
```

**검색 쿼리 예시:**

```json
GET /hotel_index/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "geo_distance": {
            "distance": "10km",
            "location": {
              "lat": 37.7749,
              "lon": -122.4194
            }
          }
        },
        {
          "range": {
            "available_dates": {
              "gte": "2025-03-01",
              "lte": "2025-03-04"
            }
          }
        }
      ],
      "filter": [
        { "term": { "amenities": "WiFi" } },
        { "range": { "room_types.min_price": { "lte": 200 } } }
      ]
    }
  },
  "sort": [
    { "star_rating": "desc" },
    { "_geo_distance": { "location": { "lat": 37.7749, "lon": -122.4194 } } }
  ],
  "from": 0,
  "size": 20
}
```

**데이터 동기화:**

MySQL (Hotel Service) → Elasticsearch 동기화는 두 가지 방식:

1. **CDC (Change Data Capture)**: Debezium으로 MySQL binlog 읽어서 실시간 동기화
2. **Batch Indexing**: 주기적으로 변경된 데이터 재색인

### 가용성 체크 최적화

검색 결과에서 "실시간" 가용성을 보여주는 것은 비용이 많이 듭니다. 두 가지 접근:

**접근 1: 검색 시 가용성 체크 안 함**
- 검색 결과는 호텔 목록만 보여줌
- 사용자가 특정 호텔 클릭 시 가용성 체크
- 장점: 검색 속도 빠름
- 단점: "Sold Out" 호텔도 결과에 나타남

**접근 2: 캐시된 가용성 정보 사용**
- Redis에 최근 가용성 정보 캐싱
- TTL 5분 설정
- 대략적인 가용성 표시 ("3개 남음", "품절 임박")
- 실제 예약 시 정확한 체크

## 객실 타입별 예약

사용자는 특정 `room_id`가 아닌 `room_type` (예: "Deluxe Double")으로 예약합니다.

### 개선된 API

```json
POST /v1/reservations
Request:
{
    "hotelID": "hotel-333",
    "roomTypeID": "deluxe-double",
    "checkInDate": "2025-03-01",
    "checkOutDate": "2025-03-04",
    "numberOfRooms": 1,
    "guestInfo": { ... },
    "idempotencyKey": "uuid-12345"
}
```

### 재고 관리

`room_type_inventory` 테이블을 사용하여 객실 타입별 재고 관리:

```sql
SELECT date, total_inventory, total_reserved
  FROM room_type_inventory
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date BETWEEN '2025-03-01' AND '2025-03-04'
```

결과:

| date | total_inventory | total_reserved |
|------|----------------|----------------|
| 2025-03-01 | 20 | 18 |
| 2025-03-02 | 20 | 17 |
| 2025-03-03 | 20 | 19 |

각 날짜에 대해 오버부킹 제한 체크:

```python
for row in results:
    if (row.total_reserved + numberOfRooms) > (row.total_inventory * 1.1):
        return "Not available"
return "Available"
```

### 재고 업데이트

예약이 확정되면 모든 날짜에 대해 재고 업데이트:

```sql
UPDATE room_type_inventory
   SET total_reserved = total_reserved + 1
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date BETWEEN '2025-03-01' AND '2025-03-04'
```

### Hot/Cold Storage 전략

- **Hot Storage (최근 1년)**: SSD 기반 MySQL, 빠른 접근
- **Cold Storage (1년 이상 과거)**: S3/Glacier, 분석 용도
- 과거 예약 데이터는 정기적으로 아카이브

## Room Assignment Flow

**핵심 질문**: room_type으로 예약하는데, 실제 room_id는 언제 할당되나?

### 문제 상황

- 사용자는 "Deluxe Double" 타입을 예약
- 호텔에는 이 타입의 객실이 20개 (room_id: 201~220)
- 예약 시점에 특정 객실을 할당하면:
  - 고객이 예약 취소/변경 시 재고 관리 복잡
  - 호텔에서 객실 유지보수/청소 상황 반영 어려움
- 체크인 직전에 할당하면:
  - 유연한 객실 관리
  - 고층 선호, 조용한 객실 등 요청 반영 가능

### 해결책: 지연된 Room Assignment

**Phase 1: 예약 생성 (room_id = NULL)**

```sql
INSERT INTO reservation (
    reservation_id, hotel_id, room_type_id, room_id,
    guest_id, check_in_date, check_out_date,
    status, created_at
) VALUES (
    'res-12345', 'hotel-333', 'deluxe-double', NULL,
    'guest-001', '2025-03-01', '2025-03-04',
    'pending', NOW()
);
```

**Phase 2: Room Assignment (체크인 24시간 전 또는 결제 완료 후)**

배치 작업 또는 이벤트 기반:

```sql
-- Step 1: 할당 가능한 객실 찾기
SELECT room_id
  FROM room
 WHERE hotel_id = 'hotel-333'
   AND room_type_id = 'deluxe-double'
   AND status = 'available'
   AND room_id NOT IN (
       SELECT room_id
         FROM reservation
        WHERE hotel_id = 'hotel-333'
          AND check_in_date < '2025-03-04'
          AND check_out_date > '2025-03-01'
          AND room_id IS NOT NULL
          AND status NOT IN ('canceled', 'checked_out')
   )
 LIMIT 1
 FOR UPDATE;

-- Step 2: 예약에 room_id 할당
UPDATE reservation
   SET room_id = '210',
       status = 'confirmed',
       updated_at = NOW()
 WHERE reservation_id = 'res-12345';
```

### Room Assignment Service

별도의 마이크로서비스로 구현:

**트리거 시점:**
1. 결제 완료 이벤트 수신 → 즉시 또는 스케줄링
2. 체크인 24시간 전 배치 작업
3. 관리자가 수동 할당

**로직:**
```python
def assign_room(reservation_id):
    reservation = get_reservation(reservation_id)

    # Special requests 고려
    preferences = parse_special_requests(reservation.special_requests)
    # 예: "High floor", "Quiet room", "Near elevator"

    # 가용 객실 찾기 (선호도 순)
    available_rooms = find_available_rooms(
        hotel_id=reservation.hotel_id,
        room_type_id=reservation.room_type_id,
        check_in=reservation.check_in_date,
        check_out=reservation.check_out_date,
        preferences=preferences
    )

    if not available_rooms:
        # Overbooking 발생: 업그레이드 또는 대체 호텔 제안
        handle_overbooking(reservation)
        return

    # 첫 번째 매칭되는 객실 할당
    assign_room_to_reservation(reservation_id, available_rooms[0].room_id)

    # 고객에게 알림
    send_notification(reservation.guest_id, "Room assigned: #" + available_rooms[0].room_number)
```

### 장단점

**장점:**
- 유연한 객실 관리 (유지보수, 청소)
- 고객 요청 반영 가능
- 오버부킹 발생 시 대응 시간 확보
- VIP 고객에게 좋은 객실 우선 할당 가능

**단점:**
- 체크인 전까지 구체적인 객실 번호 모름
- 일부 고객 불만 가능
- Room Assignment 로직 추가 복잡도

## 동시성 문제

### 문제 1: 한 사용자의 중복 예약

**시나리오**: 사용자가 "예약" 버튼을 빠르게 두 번 클릭

**해결책:**

1. **클라이언트 측 방지**
   ```javascript
   // 버튼 비활성화
   submitButton.disabled = true;
   submitButton.textContent = "Processing...";
   ```

2. **API Idempotency (멱등성)**
   ```
   POST /v1/reservations
   Header: Idempotency-Key: uuid-12345
   ```

   서버에서 idempotency_key 체크:
   ```sql
   SELECT reservation_id
     FROM reservation
    WHERE idempotency_key = 'uuid-12345';

   -- 이미 존재하면 기존 예약 반환, 없으면 새로 생성
   ```

### 문제 2: 두 사용자의 동시 예약 (Race Condition)

**시나리오**:
- 남은 객실 1개
- User A와 User B가 동시에 예약 요청
- 둘 다 "가능" 체크 통과 → 이중 예약 발생

**해결책 비교:**

#### 1. Pessimistic Locking (비관적 잠금)

```sql
BEGIN TRANSACTION;

-- Step 1: 재고 조회 + 행 잠금
SELECT date, total_inventory, total_reserved
  FROM room_type_inventory
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date BETWEEN '2025-03-01' AND '2025-03-04'
   FOR UPDATE;  -- ⬅️ 행 잠금: 다른 트랜잭션은 대기

-- Step 2: 가용성 체크
if (total_reserved + 1) > (total_inventory * 1.1):
    ROLLBACK;
    return "Not available";

-- Step 3: 재고 업데이트
UPDATE room_type_inventory
   SET total_reserved = total_reserved + 1
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date BETWEEN '2025-03-01' AND '2025-03-04';

-- Step 4: 예약 생성
INSERT INTO reservation (...) VALUES (...);

COMMIT;
```

**장점:**
- 구현 간단
- 높은 경합 상황에서 안정적 (많은 사용자가 같은 객실 예약 시도)
- 이중 예약 완전 방지

**단점:**
- 성능 저하 (다른 트랜잭션은 잠금 해제까지 대기)
- Deadlock 발생 가능 (여러 테이블 잠금 시)
- Throughput 감소

**Deadlock 방지:**
- 항상 동일한 순서로 테이블 잠금 (예: 날짜 오름차순)
- Timeout 설정 (`innodb_lock_wait_timeout = 5`)

#### 2. Optimistic Locking (낙관적 잠금)

version 컬럼 사용:

```sql
-- Step 1: 재고 조회 (잠금 없음)
SELECT date, total_inventory, total_reserved, version
  FROM room_type_inventory
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date BETWEEN '2025-03-01' AND '2025-03-04';

-- Application 레벨에서 가용성 체크
if (total_reserved + 1) > (total_inventory * 1.1):
    return "Not available";

-- Step 2: 재고 업데이트 (version 체크)
UPDATE room_type_inventory
   SET total_reserved = total_reserved + 1,
       version = version + 1
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date = '2025-03-01'
   AND version = 5;  -- ⬅️ 이전에 읽은 version

-- Step 3: 영향받은 행 수 체크
if affected_rows == 0:
    -- 다른 트랜잭션이 먼저 업데이트 → 재시도 또는 실패
    ROLLBACK;
    return "Conflict, please retry";

-- Step 4: 나머지 날짜도 동일하게 처리
-- Step 5: 예약 생성
```

**장점:**
- 잠금 없음 → 높은 throughput
- 경합이 낮을 때 (재고 많음) 효율적
- Deadlock 없음

**단점:**
- 경합이 높을 때 재시도 빈번 → 오히려 성능 저하
- Application 로직 복잡
- 사용자에게 "재시도" 메시지 노출

#### 3. Database Constraint

```sql
ALTER TABLE room_type_inventory
ADD CONSTRAINT check_room_count
CHECK (total_reserved <= total_inventory * 1.1);
```

```sql
BEGIN TRANSACTION;

UPDATE room_type_inventory
   SET total_reserved = total_reserved + 1
 WHERE room_type_id = 'deluxe-double'
   AND hotel_id = 'hotel-333'
   AND date BETWEEN '2025-03-01' AND '2025-03-04';
-- Constraint 위반 시 자동 롤백

INSERT INTO reservation (...) VALUES (...);

COMMIT;
```

**장점:**
- 구현 매우 간단
- DB 레벨 보장

**단점:**
- Constraint는 SCM 관리 어려움
- 모든 DB가 CHECK constraint 지원하는 건 아님
- 에러 메시지 불친절 (사용자에게 "Constraint violation" 노출 안 됨)
- Optimistic locking과 유사한 단점 (경합 시 롤백 빈번)

### 권장 전략

**상황별 선택:**

| 상황 | 권장 방법 | 이유 |
|-----|---------|------|
| 저 QPS (3 QPS), 일반 호텔 | Optimistic Locking | 충분한 재고, 경합 낮음 |
| 고 QPS, 인기 호텔 (재고 < 5) | Pessimistic Locking | 높은 경합, 안정성 우선 |
| 프로토타입/MVP | DB Constraint | 빠른 구현 |

**하이브리드 접근:**
```python
if inventory.total_reserved / inventory.total_inventory > 0.9:
    # 재고 10% 이하 남음 → Pessimistic locking
    use_pessimistic_locking()
else:
    # 재고 충분 → Optimistic locking
    use_optimistic_locking()
```

## 결제 통합

### 결제 플로우

```
[사용자] → [Reservation Service] → [Payment Service] → [Payment Gateway]
                                                          (Stripe, PayPal, etc.)
```

**시퀀스:**

1. **예약 생성 (status: pending)**
   ```
   POST /v1/reservations
   → reservation_id, status: pending
   ```

2. **결제 시작**
   ```
   POST /v1/payments
   Body: {
       reservationID: "res-12345",
       amount: 450.00,
       paymentMethod: "credit_card",
       cardDetails: { ... }
   }

   Payment Service → Stripe API
   ```

3. **Stripe 응답 처리**
   - 성공: payment status: completed
   - 실패: payment status: failed

4. **예약 상태 업데이트**
   - 결제 성공 → reservation status: paid
   - 결제 실패 → reservation status: canceled, 재고 복원

### 결제 실패 처리

**문제**: 재고는 차감했는데 결제 실패 → 재고 복원 필요

**해결: Timeout + Cleanup Job**

```python
# 예약 생성 시
reservation.created_at = now()
reservation.status = 'pending'

# Cleanup Job (매 5분 실행)
old_pending_reservations = db.query(
    "SELECT * FROM reservation "
    "WHERE status = 'pending' AND created_at < NOW() - INTERVAL 15 MINUTE"
)

for reservation in old_pending_reservations:
    # 재고 복원
    restore_inventory(reservation)
    # 예약 취소
    reservation.status = 'canceled'
    db.save(reservation)
```

**TTL 설정:**
- Pending 예약 유효기간: 15분
- 결제 페이지 세션 타임아웃: 10분
- 사용자에게 카운트다운 표시

### 환불 처리

```python
def process_refund(reservation_id):
    reservation = get_reservation(reservation_id)

    if reservation.status != 'paid':
        raise InvalidStateError()

    # 환불 정책 체크
    days_until_checkin = (reservation.check_in_date - today()).days

    if days_until_checkin < 1:
        refund_amount = 0  # No refund for same-day cancellation
    elif days_until_checkin < 7:
        refund_amount = reservation.total_price * 0.5  # 50% refund
    else:
        refund_amount = reservation.total_price  # Full refund

    # 결제 게이트웨이 환불 요청
    payment = get_payment(reservation_id)
    refund_response = stripe.refund(payment.transaction_id, refund_amount)

    if refund_response.success:
        # 재고 복원
        restore_inventory(reservation)

        # 상태 업데이트
        reservation.status = 'refunded'
        payment.status = 'refunded'

        # 알림 발송
        send_refund_notification(reservation.guest_id, refund_amount)

    return refund_response
```

### PCI DSS Compliance

카드 정보 처리 시 보안:

1. **카드 정보 저장 금지**: Payment Service에서 카드 번호 저장 안 함
2. **Tokenization**: Stripe 등의 게이트웨이에서 토큰 발급
3. **HTTPS**: 모든 통신 암호화
4. **Logging 주의**: 로그에 카드 정보 절대 기록 안 함

## 확장성

### Database Sharding

**Shard Key 선택: hotel_id**

이유:
- 대부분의 쿼리가 hotel_id 포함
- 호텔 간 데이터 독립적
- 균등 분산 (호텔마다 비슷한 데이터량)

**Sharding 전략:**

```
shard_id = hash(hotel_id) % num_shards
```

예: 16개 샤드, hotel_id = "hotel-333"
```python
shard_id = crc32("hotel-333") % 16  # → 5
# 5번 샤드에 저장
```

**성능 향상:**
- 30,000 QPS / 16 shards = 1,875 QPS/shard
- 각 샤드는 독립적으로 스케일

**Sharding 챌린지:**

**1. Cross-Shard Queries**

문제: "전체 호텔 검색"은 모든 샤드 조회 필요

해결책:
- **Elasticsearch 사용**: 검색은 Elasticsearch에서 처리, MySQL은 상세 정보만
- **Scatter-Gather**: 모든 샤드에 병렬 쿼리 후 결과 병합 (성능 저하 감수)

**2. Hot Shard**

문제: 특정 인기 호텔 → 해당 샤드에 트래픽 집중

해결책:
- **Caching**: 인기 호텔은 Redis 캐싱
- **Read Replica**: Hot shard에 read replica 추가
- **Consistent Hashing**: 샤드 재배치 최소화하면서 리밸런싱

**3. Rebalancing**

문제: 샤드 추가 시 데이터 재분배

해결책:
- **Consistent Hashing**: 일부 데이터만 이동
- **Virtual Nodes**: 샤드당 여러 가상 노드 (예: 16 샤드 → 256 가상 노드)
- **Blue-Green Migration**: 새 샤드 준비 후 트래픽 전환

**Sharding 미지원 쿼리:**
- "전체 예약 통계" → Data Warehouse (별도 분석 DB)
- "Top 10 인기 호텔" → 캐싱 또는 비정규화된 Summary 테이블

### Caching

**Cache Layer: Redis Cluster**

**캐싱 대상:**

1. **재고 정보** (가장 중요)
   ```
   Key: inventory:{hotel_id}:{room_type_id}:{date}
   Value: {"total": 20, "reserved": 18, "available": 2}
   TTL: 60초
   ```

2. **객실 타입 정보**
   ```
   Key: room_type:{room_type_id}
   Value: {name, description, amenities, images}
   TTL: 1시간
   ```

3. **가격 정보**
   ```
   Key: rate:{hotel_id}:{room_type_id}:{date}
   Value: 150.00
   TTL: 10분
   ```

4. **호텔 정보**
   ```
   Key: hotel:{hotel_id}
   Value: {name, location, amenities, ...}
   TTL: 1시간
   ```

**캐싱 전략:**

**Cache-Aside (Lazy Loading):**

```python
def get_inventory(hotel_id, room_type_id, date):
    cache_key = f"inventory:{hotel_id}:{room_type_id}:{date}"

    # 1. 캐시 확인
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. DB 조회
    inventory = db.query(
        "SELECT * FROM room_type_inventory "
        "WHERE hotel_id = ? AND room_type_id = ? AND date = ?",
        hotel_id, room_type_id, date
    )

    # 3. 캐시 저장
    redis.setex(cache_key, 60, json.dumps(inventory))

    return inventory
```

**Write-Through (예약 생성 시):**

```python
def create_reservation(reservation_data):
    # 1. DB 업데이트
    db.execute(
        "UPDATE room_type_inventory SET total_reserved = total_reserved + 1 ..."
    )

    # 2. 캐시 업데이트
    cache_key = f"inventory:{hotel_id}:{room_type_id}:{date}"
    redis.delete(cache_key)  # 또는 값 직접 업데이트

    # 3. 예약 레코드 생성
    db.insert_reservation(reservation_data)
```

**Cache Invalidation 전략:**

1. **TTL 기반**: 짧은 TTL 설정 (재고: 60초)
2. **Event 기반**: 예약 생성/취소 시 즉시 삭제
3. **CDC (Debezium)**: MySQL binlog 감지 → 자동 캐시 업데이트

**Cache Stampede 방지:**

문제: 인기 호텔 캐시 만료 시 동시에 수백 개 DB 쿼리

해결책:
```python
import redis_lock

def get_inventory_safe(hotel_id, room_type_id, date):
    cache_key = f"inventory:{hotel_id}:{room_type_id}:{date}"
    lock_key = f"lock:{cache_key}"

    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 락 획득 시도
    with redis_lock.Lock(redis, lock_key, expire=5):
        # 락 획득 후 다시 한번 캐시 확인 (다른 스레드가 채웠을 수 있음)
        cached = redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # DB 조회 (하나의 스레드만 실행)
        inventory = db.query(...)
        redis.setex(cache_key, 60, json.dumps(inventory))
        return inventory
```

**DB-Cache 불일치 처리:**

**시나리오**:
- DB 업데이트 성공
- 캐시 삭제 실패 (네트워크 장애)
- → 캐시에 stale data

**해결책:**
1. **짧은 TTL**: 최대 60초 불일치만 허용
2. **Retry 메커니즘**: 캐시 업데이트 실패 시 메시지 큐에 재시도 이벤트
3. **CDC**: Debezium으로 DB 변경 감지 → 캐시 자동 동기화

```
MySQL binlog → Debezium → Kafka → Cache Updater Service → Redis
```

**캐싱 장단점:**

장점:
- DB 부하 대폭 감소 (90% 이상 캐시 히트율)
- 응답 속도 빠름 (Redis: 1ms, MySQL: 10~100ms)
- 높은 QPS 처리

단점:
- 일관성 관리 복잡
- 추가 인프라 비용 (Redis 클러스터)
- 캐시 워밍 필요 (콜드 스타트 시)

## 서비스 간 데이터 일관성

### 분산 트랜잭션 문제

**시나리오: 예약 생성**

여러 서비스가 관여:
1. **Reservation Service**: 재고 차감, 예약 레코드 생성
2. **Payment Service**: 결제 처리
3. **Notification Service**: 확인 이메일 발송

문제: Payment 실패 시 Reservation 롤백 필요, 하지만 서로 다른 DB!

### 해결책: SAGA 패턴

**SAGA**: 긴 트랜잭션을 여러 로컬 트랜잭션으로 분리, 각각 보상 트랜잭션(Compensating Transaction) 정의

#### Choreography-Based SAGA (이벤트 기반)

```
[Reservation Service]
  ↓ (1) ReservationCreated 이벤트 발행
[Kafka/RabbitMQ]
  ↓ (2) 이벤트 구독
[Payment Service]
  ↓ (3) 결제 시도
  ├─ 성공 → PaymentCompleted 이벤트 발행
  │   ↓
  │  [Notification Service] → 이메일 발송
  │
  └─ 실패 → PaymentFailed 이벤트 발행
      ↓
     [Reservation Service] → 예약 취소 (보상 트랜잭션)
```

**구현 예시:**

```python
# Reservation Service
def create_reservation(data):
    # 로컬 트랜잭션 1: 재고 차감 + 예약 생성
    with db.transaction():
        update_inventory(data)
        reservation = insert_reservation(data)
        reservation.status = 'pending'

    # 이벤트 발행
    kafka.publish('reservation-events', {
        'type': 'ReservationCreated',
        'reservationID': reservation.id,
        'amount': reservation.total_price,
        'guestID': reservation.guest_id
    })

    return reservation

# 보상 트랜잭션 (결제 실패 시)
@kafka.subscribe('payment-events')
def handle_payment_failed(event):
    if event['type'] == 'PaymentFailed':
        reservation_id = event['reservationID']

        with db.transaction():
            # 재고 복원
            restore_inventory(reservation_id)
            # 예약 취소
            update_reservation_status(reservation_id, 'canceled')

        # 사용자 알림
        send_notification(event['guestID'], "Payment failed, reservation canceled")
```

```python
# Payment Service
@kafka.subscribe('reservation-events')
def handle_reservation_created(event):
    if event['type'] == 'ReservationCreated':
        reservation_id = event['reservationID']
        amount = event['amount']

        try:
            # 로컬 트랜잭션 2: 결제 시도
            payment_result = stripe.charge(amount, ...)

            with db.transaction():
                insert_payment(reservation_id, payment_result)

            # 성공 이벤트 발행
            kafka.publish('payment-events', {
                'type': 'PaymentCompleted',
                'reservationID': reservation_id,
                'paymentID': payment_result.id
            })

        except PaymentError as e:
            # 실패 이벤트 발행
            kafka.publish('payment-events', {
                'type': 'PaymentFailed',
                'reservationID': reservation_id,
                'reason': str(e)
            })
```

```python
# Notification Service
@kafka.subscribe('payment-events')
def handle_payment_completed(event):
    if event['type'] == 'PaymentCompleted':
        reservation = get_reservation(event['reservationID'])
        send_confirmation_email(reservation.guest_id, reservation)
```

**장점:**
- 각 서비스 독립적
- 낮은 결합도
- 확장 용이

**단점:**
- 플로우 추적 어려움 (분산된 이벤트)
- 순환 의존 위험
- 디버깅 복잡

#### Orchestration-Based SAGA (오케스트레이터)

중앙 Orchestrator가 전체 플로우 제어:

```
                  [Saga Orchestrator]
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
[Reservation Svc]  [Payment Svc]  [Notification Svc]
        |                |                |
        v                v                v
   Reserve Room    Process Payment   Send Email
```

**구현 예시:**

```python
# Saga Orchestrator
class BookingOrchestrator:
    def execute_booking_saga(self, booking_data):
        saga_state = {
            'step': 0,
            'reservation_id': None,
            'payment_id': None
        }

        try:
            # Step 1: 예약 생성
            saga_state['step'] = 1
            reservation = reservation_service.create(booking_data)
            saga_state['reservation_id'] = reservation.id

            # Step 2: 결제 처리
            saga_state['step'] = 2
            payment = payment_service.charge(reservation.id, reservation.total_price)
            saga_state['payment_id'] = payment.id

            # Step 3: 예약 확정
            saga_state['step'] = 3
            reservation_service.confirm(reservation.id)

            # Step 4: 알림 발송
            saga_state['step'] = 4
            notification_service.send_confirmation(reservation.guest_id, reservation)

            return {'status': 'success', 'reservation_id': reservation.id}

        except Exception as e:
            # 보상 트랜잭션 실행
            self.compensate(saga_state, e)
            return {'status': 'failed', 'reason': str(e)}

    def compensate(self, saga_state, error):
        """역순으로 보상 트랜잭션 실행"""
        if saga_state['step'] >= 2:
            # 결제 환불
            payment_service.refund(saga_state['payment_id'])

        if saga_state['step'] >= 1:
            # 예약 취소 + 재고 복원
            reservation_service.cancel(saga_state['reservation_id'])

        # 실패 알림
        notification_service.send_failure_notification(...)
```

**장점:**
- 플로우 명확 (한 곳에서 관리)
- 디버깅 쉬움
- 순서 보장

**단점:**
- Orchestrator가 SPOF (Single Point of Failure)
- 높은 결합도
- Orchestrator 복잡도 증가

### 권장: Hybrid Approach

- **핵심 플로우 (예약 → 결제)**: Orchestration (명확한 제어)
- **부가 기능 (알림, 로깅)**: Choreography (느슨한 결합)

### 기타 분산 트랜잭션 패턴

**2-Phase Commit (2PC)**:
- 강한 일관성, 하지만 성능 저하 및 가용성 문제
- 호텔 예약처럼 높은 트래픽에는 부적합
- 사용 X

**TCC (Try-Confirm/Cancel)**:
- Try: 리소스 예약 (soft lock)
- Confirm: 커밋
- Cancel: 롤백
- SAGA와 유사하지만 명시적 Try 단계
- 복잡도 높음, 필요 시만 사용

## 예약 상태 관리

### 상태 전이도

```
                    ┌──────────────┐
                    │   pending    │  (예약 요청, 재고 차감)
                    └──────┬───────┘
                           │
                  ┌────────┼────────┐
                  │                 │
            (결제 성공)        (15분 초과 or 결제 실패)
                  │                 │
                  v                 v
          ┌──────────┐         ┌──────────┐
          │   paid   │         │ canceled │ (재고 복원)
          └────┬─────┘         └──────────┘
               │
               │ (체크인)
               v
      ┌────────────────┐
      │  checked_in    │
      └───────┬────────┘
              │
              │ (체크아웃)
              v
     ┌─────────────────┐
     │  checked_out    │
     └─────────────────┘

     (취소 요청)
          ↓
     ┌──────────┐
     │ canceled │
     └─────┬────┘
           │
           │ (환불 완료)
           v
     ┌──────────┐
     │ refunded │
     └──────────┘
```

### 상태 전이 규칙

```python
ALLOWED_TRANSITIONS = {
    'pending': ['paid', 'canceled'],
    'paid': ['checked_in', 'canceled'],
    'checked_in': ['checked_out'],
    'checked_out': [],  # 종료 상태
    'canceled': ['refunded'],
    'refunded': []  # 종료 상태
}

def update_reservation_status(reservation_id, new_status):
    reservation = get_reservation(reservation_id)

    if new_status not in ALLOWED_TRANSITIONS[reservation.status]:
        raise InvalidStateTransitionError(
            f"Cannot transition from {reservation.status} to {new_status}"
        )

    # 상태별 후처리
    if new_status == 'canceled':
        restore_inventory(reservation)
    elif new_status == 'refunded':
        process_refund(reservation)

    reservation.status = new_status
    reservation.updated_at = now()
    db.save(reservation)

    # 이벤트 발행
    kafka.publish('reservation-events', {
        'type': 'ReservationStatusChanged',
        'reservationID': reservation_id,
        'oldStatus': reservation.status,
        'newStatus': new_status
    })
```

### Pending 예약 자동 정리

```python
# Cron Job: 매 5분 실행
def cleanup_expired_pending_reservations():
    expired = db.query(
        "SELECT reservation_id FROM reservation "
        "WHERE status = 'pending' "
        "AND created_at < NOW() - INTERVAL 15 MINUTE"
    )

    for reservation_id in expired:
        try:
            update_reservation_status(reservation_id, 'canceled')
            logger.info(f"Canceled expired reservation: {reservation_id}")
        except Exception as e:
            logger.error(f"Failed to cancel {reservation_id}: {e}")
```

## 모니터링 및 관찰성

### 핵심 메트릭

**비즈니스 메트릭:**
- Booking Success Rate: 전체 시도 대비 성공 예약 비율
- Average Booking Value: 평균 예약 금액
- Occupancy Rate: 날짜별 점유율
- Cancellation Rate: 취소율

**기술 메트릭:**
- **API Latency**: p50, p95, p99 응답 시간
  - Target: p99 < 500ms
- **QPS**: 초당 요청 수
- **Error Rate**: 4xx/5xx 에러 비율
  - Target: < 0.1%
- **Database Connection Pool**: 사용률
  - Alert: > 80%
- **Cache Hit Rate**: 캐시 히트율
  - Target: > 90%

**인프라 메트릭:**
- CPU/Memory/Disk 사용률
- Network I/O
- DB Replication Lag

### 로깅 전략

**구조화된 로깅 (JSON):**

```python
import logging
import json

logger = logging.getLogger(__name__)

def create_reservation(data):
    request_id = generate_request_id()

    logger.info(json.dumps({
        'event': 'reservation_creation_started',
        'request_id': request_id,
        'hotel_id': data['hotelID'],
        'room_type_id': data['roomTypeID'],
        'guest_id': data['guestID'],
        'check_in': data['checkInDate'],
        'check_out': data['checkOutDate']
    }))

    try:
        reservation = _create_reservation_internal(data)

        logger.info(json.dumps({
            'event': 'reservation_created',
            'request_id': request_id,
            'reservation_id': reservation.id,
            'status': reservation.status,
            'duration_ms': get_duration()
        }))

        return reservation

    except InventoryUnavailableError as e:
        logger.warning(json.dumps({
            'event': 'reservation_failed',
            'request_id': request_id,
            'reason': 'inventory_unavailable',
            'hotel_id': data['hotelID'],
            'room_type_id': data['roomTypeID']
        }))
        raise

    except Exception as e:
        logger.error(json.dumps({
            'event': 'reservation_error',
            'request_id': request_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
        raise
```

**로그 레벨:**
- ERROR: 시스템 오류 (DB 연결 실패, 외부 API 실패)
- WARN: 비즈니스 로직 실패 (재고 부족, 결제 실패)
- INFO: 주요 이벤트 (예약 생성, 상태 변경)
- DEBUG: 상세 디버깅 정보 (프로덕션에서는 비활성화)

**민감 정보 마스킹:**

```python
def mask_sensitive_data(data):
    if 'cardNumber' in data:
        data['cardNumber'] = '**** **** **** ' + data['cardNumber'][-4:]
    if 'cvv' in data:
        data['cvv'] = '***'
    return data
```

### 분산 추적 (Distributed Tracing)

**OpenTelemetry/Jaeger 사용:**

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("create_reservation")
def create_reservation(data):
    span = trace.get_current_span()
    span.set_attribute("hotel_id", data['hotelID'])
    span.set_attribute("room_type_id", data['roomTypeID'])

    with tracer.start_as_current_span("check_inventory"):
        inventory = check_inventory(data)

    with tracer.start_as_current_span("update_inventory"):
        update_inventory(data)

    with tracer.start_as_current_span("create_reservation_record"):
        reservation = insert_reservation(data)

    return reservation
```

**트레이스 예시:**
```
[API Gateway] create_reservation (250ms)
  ├─ [Reservation Svc] check_inventory (50ms)
  │   └─ [MySQL] SELECT inventory (45ms)
  ├─ [Reservation Svc] update_inventory (80ms)
  │   └─ [MySQL] UPDATE inventory (75ms)
  ├─ [Reservation Svc] create_reservation_record (30ms)
  └─ [Payment Svc] initiate_payment (120ms)
      └─ [Stripe API] charge (115ms)
```

### 알림 (Alerting)

**Critical Alerts (즉시 대응):**
- API Error Rate > 1% (5분간)
- p99 Latency > 2초 (5분간)
- DB Connection Pool > 95%
- Payment Service Down

**Warning Alerts (모니터링 필요):**
- Cache Hit Rate < 80%
- Booking Success Rate < 90%
- DB Replication Lag > 10초

**Alerting 도구:**
- Prometheus + Alertmanager
- Datadog
- PagerDuty (on-call 로테이션)

### 대시보드

**실시간 운영 대시보드 (Grafana):**
- 현재 QPS (API별)
- 예약 성공률 (실시간)
- 에러율 (서비스별)
- 인프라 헬스 (CPU, 메모리)

**비즈니스 대시보드:**
- 일일 예약 건수
- 매출 (일/주/월)
- Top 10 인기 호텔
- 취소율 트렌드

## 보안 고려사항

### 인증 및 인가

**JWT 기반 인증:**

```python
# 로그인 시 JWT 발급
def login(email, password):
    user = authenticate(email, password)
    if not user:
        raise AuthenticationError()

    token = jwt.encode({
        'user_id': user.id,
        'email': user.email,
        'role': user.role,  # 'guest' or 'admin'
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, SECRET_KEY, algorithm='HS256')

    return token

# API Gateway에서 토큰 검증
@require_auth
def create_reservation(request):
    token = request.headers.get('Authorization').split('Bearer ')[1]
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])

    user_id = payload['user_id']
    # ... 예약 생성 로직
```

**권한 제어 (RBAC):**

| 역할 | 권한 |
|-----|------|
| Guest | 자신의 예약 생성/조회/취소 |
| Admin | 모든 예약 조회, 호텔/객실 CRUD |
| Hotel Manager | 자신의 호텔 예약 조회, 객실 관리 |

```python
@require_role('admin')
def delete_hotel(hotel_id):
    # 관리자만 실행 가능
    pass

def cancel_reservation(reservation_id, user_id):
    reservation = get_reservation(reservation_id)

    # 본인 예약만 취소 가능
    if reservation.guest_id != user_id:
        raise PermissionDeniedError()

    # ... 취소 로직
```

### Rate Limiting

**API Gateway에서 Rate Limiting:**

```python
from redis import Redis
from datetime import timedelta

redis_client = Redis()

def rate_limit(user_id, limit=100, window=60):
    """사용자당 분당 100 요청 제한"""
    key = f"rate_limit:{user_id}"

    current = redis_client.incr(key)

    if current == 1:
        redis_client.expire(key, window)

    if current > limit:
        raise RateLimitExceededError(f"Too many requests. Limit: {limit}/min")

    return current

@app.before_request
def check_rate_limit():
    user_id = get_current_user_id()
    rate_limit(user_id)
```

**Tiered Rate Limiting:**

| 사용자 등급 | Rate Limit |
|-----------|-----------|
| Anonymous | 10 req/min |
| Registered | 100 req/min |
| Premium | 500 req/min |
| Admin | 1000 req/min |

### DDoS 방어

1. **CDN (Cloudflare, AWS CloudFront)**: Layer 3/4 DDoS 방어
2. **Web Application Firewall (WAF)**: SQL Injection, XSS 차단
3. **IP Blacklisting**: 악의적인 IP 자동 차단
4. **CAPTCHA**: 봇 방지 (의심스러운 트래픽)

### 데이터 암호화

**전송 중 (In Transit):**
- HTTPS (TLS 1.3)
- 내부 서비스 간 통신도 TLS (mTLS)

**저장 시 (At Rest):**
- Database Encryption (MySQL: InnoDB Encryption)
- S3 Bucket Encryption (AES-256)
- 민감 정보 (카드 번호) 암호화 저장

**개인정보 마스킹 (GDPR 준수):**

```python
def get_reservation_details(reservation_id, requester_user_id):
    reservation = db.get_reservation(reservation_id)

    # 본인 또는 관리자만 전체 정보 조회
    if requester_user_id != reservation.guest_id and not is_admin(requester_user_id):
        reservation.guest_email = mask_email(reservation.guest_email)
        reservation.guest_phone = mask_phone(reservation.guest_phone)

    return reservation

def mask_email(email):
    # john.doe@example.com → j***e@example.com
    local, domain = email.split('@')
    return f"{local[0]}***{local[-1]}@{domain}"
```

## 장애 처리 및 복구

### Circuit Breaker 패턴

외부 서비스 (Payment Gateway) 장애 시 cascading failure 방지:

```python
from pybreaker import CircuitBreaker

# Circuit Breaker 설정
payment_breaker = CircuitBreaker(
    fail_max=5,        # 5번 실패 시 OPEN
    timeout_duration=60  # 60초 후 HALF_OPEN
)

@payment_breaker
def charge_payment(amount, payment_details):
    response = stripe_api.charge(amount, payment_details)
    if response.status_code != 200:
        raise PaymentError()
    return response

# Circuit OPEN 시 Fallback
def create_reservation_with_fallback(data):
    try:
        reservation = create_reservation(data)
        payment = charge_payment(reservation.total_price, data['paymentDetails'])
        return reservation
    except CircuitBreakerError:
        # Payment Service가 다운 → 예약은 생성하되 결제는 나중에
        reservation = create_reservation(data)
        reservation.status = 'pending_payment'
        send_payment_link(reservation.guest_id, reservation.id)
        return reservation
```

**상태:**
- **CLOSED**: 정상 동작
- **OPEN**: 에러 임계값 초과, 요청 차단
- **HALF_OPEN**: 일부 요청 허용하여 복구 확인

### Graceful Degradation

핵심 기능은 유지하고 부가 기능 축소:

```python
def search_hotels(location, check_in, check_out):
    try:
        # Elasticsearch로 검색 (최적)
        return elasticsearch.search(location, check_in, check_out)
    except ElasticsearchError:
        # Elasticsearch 다운 → MySQL로 폴백 (느리지만 작동)
        logger.warning("Elasticsearch down, using MySQL fallback")
        return mysql_search_fallback(location, check_in, check_out)

def get_hotel_details(hotel_id):
    details = db.get_hotel(hotel_id)

    try:
        # 리뷰 데이터 추가 (외부 서비스)
        details['reviews'] = review_service.get_reviews(hotel_id, timeout=2)
    except (Timeout, ServiceUnavailableError):
        # 리뷰 서비스 다운 → 리뷰 없이 반환
        details['reviews'] = []
        logger.warning(f"Review service unavailable for hotel {hotel_id}")

    return details
```

### Retry 전략

**Exponential Backoff with Jitter:**

```python
import time
import random

def call_external_service_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff: 1초, 2초, 4초
            backoff = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Retry {attempt + 1}/{max_retries} after {backoff}s")
            time.sleep(backoff)
```

**Idempotency (재시도 안전성):**

모든 쓰기 작업은 멱등성 보장:

```python
POST /v1/reservations
Headers:
  Idempotency-Key: uuid-12345

# 서버: 동일한 Idempotency-Key로 중복 요청 시 이전 결과 반환
```

### Database Failover

**Master-Slave Replication:**

```
[Master DB] ──(replicate)──> [Slave 1]
                         └──> [Slave 2]

Write: Master
Read:  Slaves (Round-robin)
```

**Failover 시나리오:**

1. Master 다운 감지 (Health Check 실패)
2. Slave 1을 새 Master로 승격 (Automatic Failover)
3. 애플리케이션 재연결 (Connection Pool 갱신)
4. 이전 Master 복구 후 Slave로 추가

**Replication Lag 처리:**

```python
def get_reservation_after_creation(reservation_id):
    # 방금 생성한 예약 조회 → Master에서 읽기 (Read-Your-Writes)
    return db.query_master(
        "SELECT * FROM reservation WHERE reservation_id = ?",
        reservation_id
    )

def list_reservations(user_id):
    # 목록 조회 → Slave에서 읽기 (약간의 지연 허용)
    return db.query_slave(
        "SELECT * FROM reservation WHERE guest_id = ?",
        user_id
    )
```

### Backup 및 Disaster Recovery

**백업 전략:**
- **일일 풀 백업**: 매일 새벽 2시 (트래픽 최저)
- **시간별 증분 백업**: 매 시간마다
- **Binlog 백업**: 실시간 (Point-in-Time Recovery 가능)

**RTO/RPO 목표:**
- **RTO (Recovery Time Objective)**: 1시간 이내
- **RPO (Recovery Point Objective)**: 5분 이내 (Binlog로 복구)

**DR (Disaster Recovery) 시나리오:**

1. 데이터센터 전체 장애
2. 다른 Region의 Standby DB 활성화
3. DNS 업데이트 (Primary Region → DR Region)
4. 서비스 복구

## Edge Cases

### 1. 시간대 (Time Zone) 처리

**문제**: 국제 호텔, 사용자가 다른 시간대

**해결책:**
- 모든 날짜는 UTC로 저장
- 호텔마다 `timezone` 필드 (예: "America/New_York")
- 체크인/체크아웃 시간은 호텔 로컬 시간으로 표시

```python
from datetime import datetime
import pytz

def calculate_checkin_time(hotel_id, date):
    hotel = get_hotel(hotel_id)
    hotel_tz = pytz.timezone(hotel.timezone)

    # UTC 날짜 → 호텔 로컬 시간
    checkin_time = datetime.combine(date, time(15, 0))  # 3 PM
    localized = hotel_tz.localize(checkin_time)

    return localized.astimezone(pytz.UTC)  # 저장은 UTC
```

### 2. 당일 예약 (Same-Day Booking)

**문제**: 체크인 날짜 = 오늘

**고려사항:**
- 체크인 시간까지 예약 가능 (예: 오후 3시까지만)
- 빠른 확인 필요 (호텔에 즉시 알림)

```python
def validate_checkin_date(hotel_id, checkin_date):
    hotel = get_hotel(hotel_id)
    now_hotel_time = datetime.now(pytz.timezone(hotel.timezone))

    if checkin_date == now_hotel_time.date():
        # 당일 예약
        if now_hotel_time.hour >= 15:  # 3 PM 이후
            raise ValidationError("Same-day booking not available after 3 PM")
```

### 3. 다중 객실 예약

**문제**: 한 번에 3개 객실 예약

**고려사항:**
- 재고 충분한지 체크
- All-or-Nothing (3개 모두 예약 가능하거나 모두 실패)

```sql
-- 트랜잭션으로 묶기
BEGIN TRANSACTION;

SELECT total_inventory, total_reserved
  FROM room_type_inventory
 WHERE hotel_id = 'hotel-333'
   AND room_type_id = 'deluxe-double'
   AND date BETWEEN '2025-03-01' AND '2025-03-04'
   FOR UPDATE;

-- 모든 날짜에 3개 이상 가용 확인
if all_dates_have_availability(inventory, 3):
    UPDATE room_type_inventory
       SET total_reserved = total_reserved + 3
     WHERE ...;

    INSERT INTO reservation (..., number_of_rooms = 3) VALUES (...);

    COMMIT;
else:
    ROLLBACK;
```

### 4. 예약 수정

**문제**: 날짜 변경 시 재고 처리

**해결책:**
```python
def modify_reservation_dates(reservation_id, new_checkin, new_checkout):
    reservation = get_reservation(reservation_id)

    # 트랜잭션 시작
    with db.transaction():
        # 1. 기존 날짜 재고 복원
        restore_inventory(
            reservation.hotel_id,
            reservation.room_type_id,
            reservation.check_in_date,
            reservation.check_out_date,
            reservation.number_of_rooms
        )

        # 2. 새 날짜 재고 차감
        reserve_inventory(
            reservation.hotel_id,
            reservation.room_type_id,
            new_checkin,
            new_checkout,
            reservation.number_of_rooms
        )

        # 3. 예약 업데이트
        reservation.check_in_date = new_checkin
        reservation.check_out_date = new_checkout
        db.save(reservation)
```

### 5. 호텔 유지보수/폐업

**문제**: 예약된 호텔이 갑자기 폐업

**해결책:**
- Soft Delete (hotel.status = 'inactive')
- 기존 예약은 유지, 새 예약은 차단
- 영향받는 고객에게 알림 + 대체 호텔 제안

```python
def deactivate_hotel(hotel_id, reason):
    hotel = get_hotel(hotel_id)
    hotel.status = 'inactive'
    hotel.deactivation_reason = reason
    db.save(hotel)

    # 미래 예약 확인
    future_reservations = db.query(
        "SELECT * FROM reservation "
        "WHERE hotel_id = ? AND check_in_date > NOW() AND status = 'paid'",
        hotel_id
    )

    for reservation in future_reservations:
        # 고객에게 알림 + 환불 또는 대체 호텔 제안
        send_hotel_closure_notification(reservation)
```

### 6. VIP 고객 / 우선순위

**문제**: VIP 고객에게 좋은 객실 우선 할당

**해결책:**

```python
def assign_room_with_priority(reservation_id):
    reservation = get_reservation(reservation_id)
    guest = get_guest(reservation.guest_id)

    # VIP 레벨에 따라 우선순위
    priority = get_guest_priority(guest)  # 1 (VIP) ~ 5 (Regular)

    available_rooms = find_available_rooms(
        hotel_id=reservation.hotel_id,
        room_type_id=reservation.room_type_id,
        check_in=reservation.check_in_date,
        check_out=reservation.check_out_date
    )

    # 객실 점수 계산 (층수, 뷰, 크기 등)
    scored_rooms = score_rooms(available_rooms, guest.preferences)

    if priority <= 2:  # VIP
        # 최상의 객실 할당
        best_room = scored_rooms[0]
    else:
        # 랜덤 할당
        best_room = random.choice(scored_rooms)

    assign_room_to_reservation(reservation_id, best_room.room_id)
```

# 완전한 예약 플로우

## End-to-End 시퀀스 다이어그램

```
[고객]     [Frontend]  [API GW]  [Search Svc]  [Hotel Svc]  [Reservation Svc]  [Payment Svc]  [Notification Svc]
  |            |          |           |             |               |                  |                |
  |---(1) 호텔 검색------->|---------->|             |               |                  |                |
  |            |          |           |-----(Elasticsearch)          |                  |                |
  |            |<------검색 결과-------|             |               |                  |                |
  |            |          |           |             |               |                  |                |
  |---(2) 호텔 상세------->|-----------|------------>|               |                  |                |
  |            |<------호텔 정보-------|<------------|               |                  |                |
  |            |          |           |             |               |                  |                |
  |---(3) 가용성 확인----->|-----------|-------------|-------------->|                  |                |
  |            |          |           |             |<--(Redis Cache)                  |                |
  |            |<------재고 정보-------|-------------|---------------|                  |                |
  |            |          |           |             |               |                  |                |
  |---(4) 예약 요청------->|-----------|-------------|-------------->|                  |                |
  |            |          |           |             |  (Pessimistic Lock)              |                |
  |            |          |           |             |  UPDATE inventory                |                |
  |            |          |           |             |  INSERT reservation              |                |
  |            |<---reservation_id----|-------------|---------------|                  |                |
  |            |          |           |             |               |                  |                |
  |---(5) 결제 정보 입력-->|-----------|-------------|---------------|----------------->|                |
  |            |          |           |             |               |<--Stripe API---->|                |
  |            |          |           |             |               |                  |                |
  |            |          |           |  (결제 성공)                |                  |                |
  |            |          |           |             |               |<-PaymentCompleted Event           |
  |            |          |           |             |               |                  |                |
  |            |          |           |             |  UPDATE reservation.status = 'paid'              |
  |            |          |           |             |               |------------------|--------------->|
  |            |          |           |             |               |                  |  (이메일 발송) |
  |            |<---예약 확인----------|-------------|---------------|                  |                |
  |            |          |           |             |               |                  |                |
  |  (체크인 24시간 전)                             |               |                  |                |
  |            |          |           |             |  [Room Assignment Job]           |                |
  |            |          |           |             |  SELECT available room           |                |
  |            |          |           |             |  UPDATE reservation.room_id      |                |
  |            |          |           |             |               |------------------|--------------->|
  |            |          |           |             |               |                  |  (객실 배정 알림)|
  |            |          |           |             |               |                  |                |
  |---(6) 체크인---------->|-----------|-------------|---------------|                  |                |
  |            |          |           |             |  UPDATE status = 'checked_in'    |                |
  |            |<---체크인 완료--------|-------------|---------------|                  |                |
  |            |          |           |             |               |                  |                |
  |---(7) 체크아웃-------->|-----------|-------------|---------------|                  |                |
  |            |          |           |             |  UPDATE status = 'checked_out'   |                |
  |            |<---체크아웃 완료------|-------------|---------------|                  |                |
```

## 단계별 상세 설명

### 1. 호텔 검색
- 사용자: 위치, 날짜, 필터 입력
- Search Service: Elasticsearch 쿼리
- 응답: 호텔 리스트 (정렬/페이지네이션)

### 2. 호텔 상세 조회
- Hotel Service: MySQL에서 호텔 정보, 객실 타입, amenities 조회
- 이미지: CDN URL 반환

### 3. 가용성 확인
- Reservation Service:
  1. Redis 캐시 체크
  2. 캐시 미스 시 MySQL 조회
  3. 날짜별 가용 객실 수 반환

### 4. 예약 생성
- Idempotency Key 체크
- Pessimistic Locking으로 재고 조회 + 업데이트
- Reservation 레코드 생성 (status: pending)
- 15분 TTL 설정

### 5. 결제 처리
- Payment Service → Stripe API
- 성공 시: PaymentCompleted 이벤트 발행
- Reservation Service: status → paid
- Notification Service: 확인 이메일 발송

### 6. Room Assignment (체크인 24시간 전)
- 배치 작업 또는 이벤트 트리거
- 가용 객실 찾기 (특별 요청 고려)
- reservation.room_id 할당
- 고객에게 객실 번호 알림

### 7. 체크인/체크아웃
- 프론트 데스크에서 시스템 업데이트
- 또는 모바일 앱으로 Self Check-in
- 상태 변경: checked_in → checked_out

# 인터뷰 질문

## 기본 질문

**Q1: 호텔 예약 시스템을 설계하세요. Write QPS는 3입니다.**

A:
- 요구사항 확인: 기능 (예약, 검색, 취소), 비기능 (동시성, 레이턴시)
- 용량 산정: 5,000 호텔, 100만 객실, 3 QPS write, 300 QPS read
- API 설계: Hotel, Room, Reservation, Search APIs
- 데이터 모델: Hotel, Room, RoomType, Reservation, Inventory 테이블
- 아키텍처: 마이크로서비스 (Hotel, Reservation, Payment, Search, Notification)
- 동시성: Optimistic locking (QPS 낮아서)
- 확장성: Read replica, Redis 캐싱

**Q2: API를 설계하세요.**

A: RESTful APIs
- `GET /v1/hotels/search?location=&checkIn=&checkOut=`
- `GET /v1/hotels/{id}/availability`
- `POST /v1/reservations` (Idempotency-Key 포함)
- `DELETE /v1/reservations/{id}` (취소)
- 관리자용: `POST /v1/hotels`, `PUT /v1/rooms/{id}`

**Q3: 데이터 모델을 설계하세요.**

A:
- hotel: hotel_id, name, location, amenities
- room_type: room_type_id, hotel_id, type_name, max_occupancy
- room: room_id, hotel_id, room_type_id, room_number
- room_type_inventory: (hotel_id, room_type_id, date) PK, total_inventory, total_reserved
- reservation: reservation_id, hotel_id, room_type_id, room_id (nullable), guest_id, dates, status

## 동시성 질문

**Q4: 한 사용자가 중복 예약하는 문제를 어떻게 처리하나요?**

A:
- 클라이언트: 버튼 비활성화
- 서버: Idempotency Key (UUID)로 중복 요청 감지

**Q5: 두 사용자가 마지막 객실을 동시에 예약하는 경쟁 상태를 어떻게 해결하나요?**

A: 세 가지 방법
1. **Pessimistic Locking**: `SELECT ... FOR UPDATE`, 높은 경합 시 적합
2. **Optimistic Locking**: version 컬럼, 낮은 경합 시 적합
3. **Database Constraint**: CHECK constraint, 간단하지만 에러 처리 어려움

권장: 재고 10% 이하 남으면 Pessimistic, 그 외 Optimistic

**Q6: RDBMS Isolation Level을 설명하세요.**

A:
- READ UNCOMMITTED: Dirty Read 가능
- READ COMMITTED: Committed 데이터만 읽기, Non-repeatable Read 발생
- REPEATABLE READ: 트랜잭션 내 일관된 읽기, Phantom Read 가능
- SERIALIZABLE: 완전 격리, 성능 저하

호텔 예약은 REPEATABLE READ 사용 (MySQL InnoDB 기본값)

**Q7: MySQL Lock 종류를 설명하세요.**

A:
- Shared Lock (S): 읽기 잠금, 여러 트랜잭션 동시 획득 가능
- Exclusive Lock (X): 쓰기 잠금, 하나만 획득
- Record Lock: 인덱스 레코드 잠금
- Gap Lock: 레코드 간 간격 잠금 (Phantom Read 방지)
- Next-Key Lock: Record + Gap Lock 조합
- Intention Lock: 테이블 레벨 의도 잠금

## 확장성 질문

**Q8: 객실 타입별 예약을 어떻게 설계하나요?**

A:
- room_type_inventory 테이블 사용
- (hotel_id, room_type_id, date)로 날짜별 재고 추적
- 예약 시 room_id는 NULL, 나중에 할당
- Room Assignment Service가 체크인 전에 특정 객실 배정

**Q9: QPS가 Booking.com처럼 1,000배 높아지면?**

A:
1. **Database Sharding**: hotel_id로 샤드 분할 (16 샤드 → 1,875 QPS/shard)
2. **Caching**: Redis에 재고 정보 캐싱 (TTL 60초), Cache-aside + CDC
3. **Read Replica**: 읽기 트래픽 분산
4. **CDN**: 정적 콘텐츠 (이미지)
5. **Elasticsearch**: 검색 트래픽 분리
6. **Horizontal Scaling**: 각 마이크로서비스 Auto Scaling

**Q10: 분산 트랜잭션을 어떻게 처리하나요?**

A:
- **SAGA 패턴** 사용 (Orchestration-based)
- 예약 생성 → 결제 → 알림 (각각 로컬 트랜잭션)
- 결제 실패 시 보상 트랜잭션: 예약 취소 + 재고 복원
- Kafka로 이벤트 발행/구독
- 2PC는 성능 문제로 사용 X

## 심화 질문

**Q11: 높은 동시성을 어떻게 처리하나요?**

A:
- Application 레벨: Optimistic/Pessimistic locking
- Database 레벨: 적절한 Isolation Level (REPEATABLE READ)
- Cache 레벨: Redis 분산 락 (재고 업데이트 시)
- Message Queue: 비동기 처리 (알림 발송)
- Rate Limiting: API Gateway에서 과도한 요청 차단

**Q12: 오버부킹 방지 전략은?**

A:
- room_type_inventory 테이블에서 실시간 체크
- 조건: `total_reserved + numberOfRooms <= total_inventory * 1.1`
- DB Constraint: `CHECK (total_reserved <= total_inventory * 1.1)`
- Pessimistic locking으로 원자성 보장
- 만약 오버부킹 발생 시: 객실 업그레이드 또는 대체 호텔 제안

**Q13: Database Sharding 전략은? 확장성은?**

A:
- Shard Key: hotel_id (대부분 쿼리에 포함, 균등 분산)
- Hash-based: `shard_id = hash(hotel_id) % num_shards`
- 장점: 수평 확장, 각 샤드 독립적
- 챌린지:
  - Cross-shard 쿼리 → Elasticsearch 사용
  - Hot shard → Caching + Read Replica
  - Rebalancing → Consistent Hashing

**Q14: 캐싱 메커니즘은?**

A:
- Redis Cluster 사용
- 재고 정보: `inventory:{hotel_id}:{room_type_id}:{date}`, TTL 60초
- 가격 정보: TTL 10분
- 호텔 정보: TTL 1시간
- Cache-aside + CDC (Debezium)로 DB-Cache 동기화
- Cache Stampede 방지: 분산 락

**Q15: 서비스 간 데이터 일관성은?**

A:
- SAGA 패턴 (Orchestration-based)
- 예: 예약 → 결제 → 알림
- 각 단계는 로컬 트랜잭션
- 실패 시 보상 트랜잭션 (역순 실행)
- Kafka로 이벤트 기반 통신
- Orchestrator가 전체 플로우 관리

**Q16: room_type_inventory 역할은?**

A:
- 객실 타입별 날짜별 재고 추적
- total_inventory: 해당 타입의 전체 객실 수
- total_reserved: 예약된 객실 수
- 빠른 가용성 체크: `total_inventory - total_reserved`
- 인덱스: (hotel_id, room_type_id, date)로 범위 쿼리 최적화
- 캐싱 가능 (Redis)

**Q17: 객실 타입별 예약 API 변경사항은?**

A:
- 기존: `roomID` (특정 객실) → 개선: `roomTypeID` (객실 타입)
- 장점:
  - 사용자 경험 향상 (구체적인 객실 번호 몰라도 됨)
  - 유연한 객실 관리 (호텔에서 최적 배정)
  - 오버부킹 발생 시 대체 가능
- room_id는 체크인 전 Room Assignment Service가 할당

**Q18: 트래픽 1,000배 증가 시 대응은?**

A:
1. Sharding: 쓰기 부하 분산
2. Caching: 읽기 부하 대폭 감소
3. Read Replica: 읽기 쿼리 분산
4. CDN: 정적 콘텐츠 오프로드
5. Auto Scaling: 애플리케이션 서버 동적 확장
6. Load Balancer: 트래픽 분산
7. Message Queue: 비동기 처리 (알림, 로깅)
8. Monitoring: Prometheus + Grafana로 병목 지점 실시간 감지

**Q19: Database Constraint의 단점은?**

A:
- SCM 관리 어려움 (코드와 분리된 DB 스키마)
- 모든 DB가 CHECK constraint 지원하는 건 아님 (MySQL 8.0.16+)
- 에러 메시지 불친절 (사용자에게 "Constraint violation" 노출 안 됨)
- 경합 높을 때 빈번한 롤백 → 성능 저하 (Optimistic locking과 유사)
- Application 로직에서 제어 불가 (유연성 부족)

**Q20: 두 사용자가 마지막 객실을 동시 예약 시 해결은?**

A:
- **Pessimistic Locking** 사용:
  1. User A: `SELECT ... FOR UPDATE` → 행 잠금 획득
  2. User B: 동일 쿼리 → 대기
  3. User A: 재고 체크 → 업데이트 → 커밋
  4. User B: 잠금 획득 → 재고 체크 → 이미 0개 → 실패
- 또는 **Optimistic Locking**:
  1. User A, B: 동시에 재고 조회 (version = 5)
  2. User A: `UPDATE ... WHERE version = 5` → 성공, version = 6
  3. User B: `UPDATE ... WHERE version = 5` → 실패 (affected_rows = 0)
  4. User B: 재시도 또는 "Sold Out" 응답

## 운영 질문

**Q21: 모니터링 전략은?**

A:
- **메트릭**: Booking Success Rate, p99 Latency, QPS, Error Rate, Cache Hit Rate
- **로깅**: 구조화된 JSON 로그 (ELK Stack)
- **Tracing**: OpenTelemetry/Jaeger로 분산 추적
- **알림**: Prometheus Alertmanager (Error Rate > 1%, Latency > 2초)
- **대시보드**: Grafana (실시간 QPS, 예약 성공률, 인프라 헬스)

**Q22: 보안 고려사항은?**

A:
- **인증**: JWT 기반
- **인가**: RBAC (Guest, Admin, Hotel Manager)
- **Rate Limiting**: API Gateway에서 사용자당 100 req/min
- **DDoS 방어**: CDN (Cloudflare), WAF
- **암호화**: HTTPS (TLS 1.3), DB Encryption at Rest
- **민감 정보**: 카드 번호 Tokenization (Stripe), 로그 마스킹
- **GDPR**: 개인정보 마스킹, 삭제 요청 처리

**Q23: 장애 처리 전략은?**

A:
- **Circuit Breaker**: 외부 서비스 (Payment) 장애 시 fallback
- **Graceful Degradation**: 검색 서비스 다운 시 MySQL로 폴백
- **Retry**: Exponential backoff with jitter
- **Idempotency**: 재시도 시 중복 방지
- **Database Failover**: Master 다운 시 Slave 자동 승격
- **Backup**: 일일 풀 백업 + 시간별 증분 + Binlog

**Q24: Edge Cases는?**

A:
- **Time Zone**: 모든 날짜 UTC 저장, 호텔별 timezone 필드로 변환
- **당일 예약**: 체크인 시간(3 PM)까지만 허용
- **다중 객실**: All-or-nothing 트랜잭션
- **예약 수정**: 기존 재고 복원 + 새 재고 차감 (원자적)
- **호텔 폐업**: Soft delete, 기존 예약 유지, 대체 호텔 제안
- **VIP 고객**: 우선순위 기반 Room Assignment

# References

* [System Design Interview Volume 2 - Chapter 18: Hotel Reservation System](https://www.amazon.com/System-Design-Interview-Insiders-Guide/dp/1736049119)
* [Booking.com Engineering Blog](https://blog.booking.com/)
* [Airbnb Engineering Blog](https://medium.com/airbnb-engineering)
* [RestAppHotelbooking | java](https://github.com/BogushAleksandr/RestAppHotelbooking)
* [Booking Application | go](https://github.com/revel/examples/blob/master/README.md)
* [Hotel Reservation Management System | python](https://github.com/rub9542/Hotel-Reservation-Management-System)
* [MySQL Locking](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html)
* [SAGA Pattern](https://microservices.io/patterns/data/saga.html)
* [Debezium CDC](https://debezium.io/)
