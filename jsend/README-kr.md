# JSend - 간단한 JSON 응답 규칙

## 목차
- [JSend란?](#jsend란)
- [JSend의 3가지 상태](#jsend의-3가지-상태)
- [JSend + Google JSON Guide 결합](#jsend--google-json-guide-결합)
- [ETag 완전 정복](#etag-완전-정복)
- [권장 최종 포맷 예제 JSON](#권장-최종-포맷-예제-json)
- [권장 최종 포맷 예제 코드](#권장-최종-포맷-예제-코드)

----

## JSend란?

**JSend**는 서버가 클라이언트에게 JSON으로 응답할 때 사용하는 **초간단 규칙**입니다.

### 핵심 개념

"API 응답은 항상 같은 형태로 보내자!"

- **제작**: OmniTI (미국 웹 인프라 회사)
- **공식 사이트**: https://github.com/omniti-labs/jsend
- **철학**: 극도로 단순하고 실용적

### 왜 필요한가?

**문제 상황:**
```json
// API A의 응답
{ "result": { ... } }

// API B의 응답
{ "data": { ... } }

// API C의 응답
{ "response": { ... } }
```

모든 API가 다른 형식을 쓰면 혼란스럽습니다!

**JSend 해결책:**
```json
// 모든 API가 동일한 형식 사용
{
  "status": "success",
  "data": { ... }
}
```

일관성 있고 예측 가능합니다!

----

## JSend의 3가지 상태

JSend는 딱 **3가지 상태**만 있습니다.

### 1. success (성공)

요청이 성공했을 때

```json
{
  "status": "success",
  "data": {
    "userId": "12345",
    "userName": "홍길동"
  }
}
```

**규칙:**
- `status`는 반드시 `"success"`
- `data`에 실제 데이터를 넣음
- HTTP 상태 코드: 2xx (200, 201 등)

### 2. fail (클라이언트 오류)

사용자가 잘못된 요청을 보냈을 때

```json
{
  "status": "fail",
  "data": {
    "email": "이메일 형식이 올바르지 않습니다",
    "password": "비밀번호는 최소 8자 이상이어야 합니다"
  }
}
```

**규칙:**
- `status`는 반드시 `"fail"`
- `data`에 어떤 필드가 왜 실패했는지 설명
- HTTP 상태 코드: 4xx (400, 403, 404 등)

### 3. error (서버 오류)

서버에서 문제가 생겼을 때

```json
{
  "status": "error",
  "message": "데이터베이스 연결 실패",
  "code": "DB_CONNECTION_ERROR"
}
```

**규칙:**
- `status`는 반드시 `"error"`
- `message`는 필수 (에러 설명)
- `code`, `data`는 선택 사항
- HTTP 상태 코드: 5xx (500, 503 등)

### fail vs error 비교

| 구분 | fail | error |
|------|------|-------|
| **누구 잘못?** | 클라이언트 (사용자) | 서버 |
| **예시** | 비밀번호 틀림, 필수 항목 누락 | DB 오류, 서버 다운 |
| **HTTP 코드** | 4xx | 5xx |
| **해결 방법** | 사용자가 입력 수정 | 개발자가 서버 수정 |

----

## JSend + Google JSON Guide 결합

JSend는 단순하지만 실무에서는 **추가 정보**가 필요합니다.
Google JSON Guide의 유용한 부분을 가져와 결합하면 완벽합니다!

### Google JSON Guide의 유용한 개념

1. **selfLink**: 리소스의 URL (HATEOAS)
2. **etag**: 캐싱용 버전 태그
3. **pagination**: 페이지네이션 정보
4. **meta**: 메타데이터 분리

### 결합 전략

**JSend 기본 구조 유지 + Google의 유용한 부분 추가**

```json
{
  "status": "success",      // JSend 핵심
  "data": { ... },          // JSend 핵심
  "meta": { ... }           // Google의 메타데이터 추가
}
```

### 권장 최종 구조

```json
{
  "status": "success | fail | error",
  "data": {
    // 실제 데이터
    "id": "resource-123",
    "selfLink": "/api/v1/resources/resource-123"  // Google 스타일
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "etag": "\"abc123\"",
    "pagination": {  // 목록 조회 시
      "page": 1,
      "pageSize": 20,
      "totalItems": 156,
      "totalPages": 8
    }
  },
  "message": "...",  // fail, error 시 사용
  "code": "..."      // error 시 사용
}
```

----

## ETag 완전 정복

### ETag란?

**ETag(Entity Tag)**는 데이터의 **"버전 번호"** 같은 것입니다.
서버의 데이터가 바뀌었는지 확인하는 짧은 문자열 태그입니다.

### 쉬운 비유

**도서관에서 책 빌리는 상황:**

1. **첫 방문**: "해리포터" 책을 빌림
   - 사서: "이 책은 2023년 3월 개정판이에요" (ETag: "v2023.03")

2. **1주일 후**: 같은 책을 또 빌리러 감
   - 나: "v2023.03 버전인데, 새 버전 나왔어요?"
   - 사서: "아니요, 여전히 v2023.03이에요!"
   - 나: "그럼 집에 있는 책 그대로 읽을게요!"

3. **1달 후**: 또 방문
   - 나: "v2023.03 버전인데, 새 버전 나왔어요?"
   - 사서: "네! v2024.01 개정판이 나왔어요!"
   - 나: "새 버전으로 빌려갈게요!"

### 왜 필요한가?

**문제 (ETag 없이):**
- 불필요한 데이터 전송 (3MB × 100번 = 300MB)
- 느린 응답 속도
- 서버 부하 증가

**해결 (ETag 사용):**
- 데이터 전송 최소화 (3MB → 0MB)
- 빠른 응답 (2초 → 0.1초)
- 서버 부하 감소

### 작동 원리

#### 1단계: 첫 요청

**요청:**
```http
GET /api/v1/hotels/hotel-123
```

**응답:**
```http
HTTP/1.1 200 OK
ETag: "abc123"

{
  "status": "success",
  "data": {
    "id": "hotel-123",
    "name": "그랜드 호텔"
  },
  "meta": {
    "etag": "abc123"
  }
}
```

#### 2단계: 다시 요청

**요청 (ETag 포함):**
```http
GET /api/v1/hotels/hotel-123
If-None-Match: "abc123"
```

**응답 A (데이터 안 바뀜):**
```http
HTTP/1.1 304 Not Modified
ETag: "abc123"
```

**응답 B (데이터 바뀜):**
```http
HTTP/1.1 200 OK
ETag: "xyz789"

{
  "status": "success",
  "data": {
    "id": "hotel-123",
    "name": "그랜드 호텔",
    "rating": 4.8  // 변경됨!
  },
  "meta": {
    "etag": "xyz789"
  }
}
```

### 성능 비교

| 항목 | ETag 없음 | ETag 사용 |
|------|-----------|-----------|
| 데이터 전송량 | 3MB × 100회 = 300MB | 3MB × 1회 = 3MB |
| 평균 응답 시간 | 2초 | 0.1초 |
| 서버 CPU 사용률 | 높음 (100%) | 낮음 (20%) |
| 사용자 경험 | 느림 | 빠름 |

----

## 권장 최종 포맷 예제 JSON

### 1. 단순 조회 (성공)

```json
GET /api/v1/hotels/hotel-123

Response (200 OK):
{
  "status": "success",
  "data": {
    "id": "hotel-123",
    "name": "그랜드 호텔",
    "rating": 4.5,
    "address": "서울시 강남구",
    "selfLink": "/api/v1/hotels/hotel-123"
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "etag": "\"3a-4b-5c\""
  }
}
```

### 2. 목록 조회 (페이지네이션)

```json
GET /api/v1/hotels?page=1&pageSize=20

Response (200 OK):
{
  "status": "success",
  "data": {
    "items": [
      {
        "id": "hotel-123",
        "name": "그랜드 호텔",
        "rating": 4.5,
        "selfLink": "/api/v1/hotels/hotel-123"
      },
      {
        "id": "hotel-456",
        "name": "럭셔리 호텔",
        "rating": 4.8,
        "selfLink": "/api/v1/hotels/hotel-456"
      }
    ],
    "pagination": {
      "page": 1,
      "pageSize": 20,
      "totalItems": 156,
      "totalPages": 8,
      "hasNext": true,
      "hasPrev": false,
      "nextPage": "/api/v1/hotels?page=2",
      "prevPage": null
    }
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 3. 생성 (성공)

```json
POST /api/v1/hotels

Request:
{
  "name": "새로운 호텔",
  "address": "서울시 종로구",
  "rating": 4.0
}

Response (201 Created):
{
  "status": "success",
  "data": {
    "id": "hotel-789",
    "name": "새로운 호텔",
    "address": "서울시 종로구",
    "rating": 4.0,
    "createdAt": "2024-01-15T10:30:00Z",
    "selfLink": "/api/v1/hotels/hotel-789"
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "etag": "\"1a-2b-3c\""
  }
}
```

### 4. 유효성 검증 실패

```json
POST /api/v1/hotels

Response (400 Bad Request):
{
  "status": "fail",
  "message": "입력 데이터 검증 실패",
  "data": {
    "errors": [
      {
        "field": "name",
        "reason": "required",
        "message": "호텔 이름은 필수입니다"
      },
      {
        "field": "rating",
        "reason": "outOfRange",
        "message": "평점은 0~5 사이여야 합니다",
        "value": 6
      }
    ]
  }
}
```

### 5. 권한 없음

```json
DELETE /api/v1/hotels/hotel-123

Response (403 Forbidden):
{
  "status": "fail",
  "message": "권한이 없습니다",
  "data": {
    "authorization": "이 작업은 관리자만 수행할 수 있습니다",
    "requiredRole": "ADMIN",
    "currentRole": "USER"
  }
}
```

### 6. 리소스 없음

```json
GET /api/v1/hotels/hotel-999

Response (404 Not Found):
{
  "status": "fail",
  "message": "리소스를 찾을 수 없습니다",
  "data": {
    "resourceType": "hotel",
    "resourceId": "hotel-999",
    "reason": "호텔 ID 'hotel-999'는 존재하지 않습니다"
  }
}
```

### 7. 서버 오류

```json
GET /api/v1/hotels

Response (500 Internal Server Error):
{
  "status": "error",
  "message": "데이터베이스 연결 실패",
  "code": "DB_CONNECTION_ERROR",
  "data": {
    "timestamp": "2024-01-15T10:30:00Z",
    "traceId": "abc-123-def-456"
  }
}
```

----

## 권장 최종 포맷 예제 코드

### Server: Java Spring MVC

#### ApiResponse.java (공통 응답 DTO)

```java
package com.example.demo.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiResponse<T> {
    private String status;      // success, fail, error
    private T data;             // 응답 데이터
    private MetaData meta;      // 메타데이터
    private String message;     // fail, error 시 메시지
    private String code;        // error 시 코드

    // Success 응답 생성
    public static <T> ApiResponse<T> success(T data) {
        return ApiResponse.<T>builder()
            .status("success")
            .data(data)
            .meta(MetaData.builder()
                .timestamp(java.time.Instant.now().toString())
                .build())
            .build();
    }

    // Success with ETag 응답 생성
    public static <T> ApiResponse<T> success(T data, String etag) {
        return ApiResponse.<T>builder()
            .status("success")
            .data(data)
            .meta(MetaData.builder()
                .timestamp(java.time.Instant.now().toString())
                .etag(etag)
                .build())
            .build();
    }

    // Fail 응답 생성
    public static <T> ApiResponse<T> fail(String message, T data) {
        return ApiResponse.<T>builder()
            .status("fail")
            .message(message)
            .data(data)
            .build();
    }

    // Error 응답 생성
    public static <T> ApiResponse<T> error(String message, String code) {
        return ApiResponse.<T>builder()
            .status("error")
            .message(message)
            .code(code)
            .build();
    }
}

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
class MetaData {
    private String timestamp;
    private String etag;
    private PaginationInfo pagination;
}

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
class PaginationInfo {
    private int page;
    private int pageSize;
    private long totalItems;
    private int totalPages;
    private boolean hasNext;
    private boolean hasPrev;
    private String nextPage;
    private String prevPage;
}
```

#### HotelController.java

```java
package com.example.demo.controller;

import com.example.demo.dto.*;
import com.example.demo.model.Hotel;
import com.example.demo.service.HotelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/hotels")
public class HotelController {

    @Autowired
    private HotelService hotelService;

    // 1. 단일 호텔 조회 (with ETag)
    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<HotelResponse>> getHotel(
        @PathVariable String id,
        @RequestHeader(value = "If-None-Match", required = false) String ifNoneMatch
    ) {
        try {
            // 호텔 조회
            Hotel hotel = hotelService.findById(id);

            if (hotel == null) {
                // 404 Not Found
                Map<String, String> failData = new HashMap<>();
                failData.put("resourceType", "hotel");
                failData.put("resourceId", id);
                failData.put("reason", "호텔 ID '" + id + "'는 존재하지 않습니다");

                ApiResponse<Map<String, String>> response =
                    ApiResponse.fail("리소스를 찾을 수 없습니다", failData);

                return ResponseEntity
                    .status(HttpStatus.NOT_FOUND)
                    .body((ApiResponse) response);
            }

            // ETag 생성
            String etag = generateETag(hotel);

            // ETag 비교 - 304 Not Modified
            if (etag.equals(ifNoneMatch)) {
                return ResponseEntity
                    .status(HttpStatus.NOT_MODIFIED)
                    .eTag(etag)
                    .build();
            }

            // 200 OK with data
            HotelResponse hotelResponse = HotelResponse.fromEntity(hotel);
            ApiResponse<HotelResponse> response = ApiResponse.success(hotelResponse, etag);

            return ResponseEntity
                .ok()
                .eTag(etag)
                .body(response);

        } catch (Exception e) {
            // 500 Internal Server Error
            ApiResponse<Void> response = ApiResponse.error(
                "서버 내부 오류가 발생했습니다",
                "INTERNAL_SERVER_ERROR"
            );
            return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body((ApiResponse) response);
        }
    }

    // 2. 호텔 목록 조회 (with Pagination)
    @GetMapping
    public ResponseEntity<ApiResponse<HotelListResponse>> getHotels(
        @RequestParam(defaultValue = "1") int page,
        @RequestParam(defaultValue = "20") int pageSize
    ) {
        try {
            // 호텔 목록 조회
            List<Hotel> hotels = hotelService.findAll(page, pageSize);
            long totalItems = hotelService.count();
            int totalPages = (int) Math.ceil((double) totalItems / pageSize);

            // Response DTO 변환
            List<HotelResponse> items = hotels.stream()
                .map(HotelResponse::fromEntity)
                .collect(Collectors.toList());

            // Pagination 정보 구성
            PaginationInfo pagination = PaginationInfo.builder()
                .page(page)
                .pageSize(pageSize)
                .totalItems(totalItems)
                .totalPages(totalPages)
                .hasNext(page < totalPages)
                .hasPrev(page > 1)
                .nextPage(page < totalPages ? "/api/v1/hotels?page=" + (page + 1) : null)
                .prevPage(page > 1 ? "/api/v1/hotels?page=" + (page - 1) : null)
                .build();

            HotelListResponse data = HotelListResponse.builder()
                .items(items)
                .pagination(pagination)
                .build();

            ApiResponse<HotelListResponse> response = ApiResponse.success(data);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            ApiResponse<Void> response = ApiResponse.error(
                "호텔 목록 조회 실패",
                "HOTEL_LIST_ERROR"
            );
            return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body((ApiResponse) response);
        }
    }

    // 3. 호텔 생성
    @PostMapping
    public ResponseEntity<ApiResponse<HotelResponse>> createHotel(
        @RequestBody @Valid HotelCreateRequest request,
        BindingResult bindingResult
    ) {
        try {
            // 유효성 검증 실패
            if (bindingResult.hasErrors()) {
                List<ValidationError> errors = bindingResult.getFieldErrors().stream()
                    .map(error -> ValidationError.builder()
                        .field(error.getField())
                        .reason("validation")
                        .message(error.getDefaultMessage())
                        .value(error.getRejectedValue())
                        .build())
                    .collect(Collectors.toList());

                Map<String, Object> failData = new HashMap<>();
                failData.put("errors", errors);

                ApiResponse<Map<String, Object>> response =
                    ApiResponse.fail("입력 데이터 검증 실패", failData);

                return ResponseEntity
                    .status(HttpStatus.BAD_REQUEST)
                    .body((ApiResponse) response);
            }

            // 호텔 생성
            Hotel hotel = hotelService.create(request);
            HotelResponse hotelResponse = HotelResponse.fromEntity(hotel);

            // ETag 생성
            String etag = generateETag(hotel);

            ApiResponse<HotelResponse> response = ApiResponse.success(hotelResponse, etag);

            return ResponseEntity
                .status(HttpStatus.CREATED)
                .eTag(etag)
                .body(response);

        } catch (Exception e) {
            ApiResponse<Void> response = ApiResponse.error(
                "호텔 생성 실패",
                "HOTEL_CREATE_ERROR"
            );
            return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body((ApiResponse) response);
        }
    }

    // 4. 호텔 삭제
    @DeleteMapping("/{id}")
    public ResponseEntity<ApiResponse<Map<String, Object>>> deleteHotel(
        @PathVariable String id
    ) {
        try {
            // 호텔 존재 여부 확인
            Hotel hotel = hotelService.findById(id);

            if (hotel == null) {
                Map<String, String> failData = new HashMap<>();
                failData.put("resourceType", "hotel");
                failData.put("resourceId", id);
                failData.put("reason", "호텔 ID '" + id + "'는 존재하지 않습니다");

                ApiResponse<Map<String, String>> response =
                    ApiResponse.fail("리소스를 찾을 수 없습니다", failData);

                return ResponseEntity
                    .status(HttpStatus.NOT_FOUND)
                    .body((ApiResponse) response);
            }

            // 호텔 삭제
            hotelService.delete(id);

            Map<String, Object> data = new HashMap<>();
            data.put("id", id);
            data.put("deleted", true);
            data.put("deletedAt", java.time.Instant.now().toString());

            ApiResponse<Map<String, Object>> response = ApiResponse.success(data);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            ApiResponse<Void> response = ApiResponse.error(
                "호텔 삭제 실패",
                "HOTEL_DELETE_ERROR"
            );
            return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body((ApiResponse) response);
        }
    }

    // ETag 생성 헬퍼 메서드
    private String generateETag(Hotel hotel) {
        try {
            String data = hotel.getId() + hotel.getUpdatedAt().toString();
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(data.getBytes());
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return "\"" + sb.toString() + "\"";
        } catch (Exception e) {
            return "\"" + System.currentTimeMillis() + "\"";
        }
    }
}
```

#### HotelResponse.java (Response DTO)

```java
package com.example.demo.dto;

import com.example.demo.model.Hotel;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class HotelResponse {
    private String id;
    private String name;
    private String address;
    private Double rating;
    private String selfLink;

    public static HotelResponse fromEntity(Hotel hotel) {
        return HotelResponse.builder()
            .id(hotel.getId())
            .name(hotel.getName())
            .address(hotel.getAddress())
            .rating(hotel.getRating())
            .selfLink("/api/v1/hotels/" + hotel.getId())
            .build();
    }
}

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
class HotelListResponse {
    private List<HotelResponse> items;
    private PaginationInfo pagination;
}

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
class ValidationError {
    private String field;
    private String reason;
    private String message;
    private Object value;
}
```

#### HotelCreateRequest.java (Request DTO)

```java
package com.example.demo.dto;

import lombok.Data;
import javax.validation.constraints.*;

@Data
public class HotelCreateRequest {
    @NotBlank(message = "호텔 이름은 필수입니다")
    @Size(min = 2, max = 100, message = "호텔 이름은 2~100자 사이여야 합니다")
    private String name;

    @NotBlank(message = "주소는 필수입니다")
    private String address;

    @NotNull(message = "평점은 필수입니다")
    @Min(value = 0, message = "평점은 0 이상이어야 합니다")
    @Max(value = 5, message = "평점은 5 이하여야 합니다")
    private Double rating;
}
```

### Client: TypeScript + React

#### api.ts (API Client with ETag)

```typescript
// api.ts
import axios, { AxiosInstance, AxiosResponse } from 'axios';

// API Response 타입
export interface ApiResponse<T = any> {
  status: 'success' | 'fail' | 'error';
  data?: T;
  meta?: MetaData;
  message?: string;
  code?: string;
}

interface MetaData {
  timestamp?: string;
  etag?: string;
  pagination?: PaginationInfo;
}

interface PaginationInfo {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
  nextPage: string | null;
  prevPage: string | null;
}

// Hotel 타입
export interface Hotel {
  id: string;
  name: string;
  address: string;
  rating: number;
  selfLink: string;
}

export interface HotelListData {
  items: Hotel[];
  pagination: PaginationInfo;
}

export interface HotelCreateRequest {
  name: string;
  address: string;
  rating: number;
}

// ETag 캐시
class ETagCache {
  private cache: Map<string, { etag: string; data: any }> = new Map();

  set(url: string, etag: string, data: any): void {
    this.cache.set(url, { etag, data });
  }

  get(url: string): { etag: string; data: any } | undefined {
    return this.cache.get(url);
  }

  clear(): void {
    this.cache.clear();
  }
}

// API Client 클래스
class ApiClient {
  private axiosInstance: AxiosInstance;
  private etagCache: ETagCache;

  constructor(baseURL: string) {
    this.etagCache = new ETagCache();
    this.axiosInstance = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request Interceptor: ETag 추가
    this.axiosInstance.interceptors.request.use((config) => {
      const cached = this.etagCache.get(config.url || '');
      if (cached && config.method?.toLowerCase() === 'get') {
        config.headers['If-None-Match'] = cached.etag;
      }
      return config;
    });

    // Response Interceptor: ETag 처리
    this.axiosInstance.interceptors.response.use(
      (response) => {
        // ETag 저장
        const etag = response.headers['etag'];
        if (etag && response.config.url) {
          this.etagCache.set(response.config.url, etag, response.data);
        }
        return response;
      },
      (error) => {
        // 304 Not Modified 처리
        if (error.response?.status === 304) {
          const cached = this.etagCache.get(error.config.url);
          if (cached) {
            console.log('✅ 304 Not Modified - 캐시 사용');
            return Promise.resolve({
              ...error.response,
              data: cached.data,
              fromCache: true,
            });
          }
        }
        return Promise.reject(error);
      }
    );
  }

  // GET 요청
  async get<T = any>(url: string): Promise<ApiResponse<T>> {
    const response: AxiosResponse<ApiResponse<T>> = await this.axiosInstance.get(url);
    return response.data;
  }

  // POST 요청
  async post<T = any>(url: string, data: any): Promise<ApiResponse<T>> {
    const response: AxiosResponse<ApiResponse<T>> = await this.axiosInstance.post(url, data);
    return response.data;
  }

  // PUT 요청
  async put<T = any>(url: string, data: any): Promise<ApiResponse<T>> {
    const response: AxiosResponse<ApiResponse<T>> = await this.axiosInstance.put(url, data);
    return response.data;
  }

  // DELETE 요청
  async delete<T = any>(url: string): Promise<ApiResponse<T>> {
    const response: AxiosResponse<ApiResponse<T>> = await this.axiosInstance.delete(url);
    return response.data;
  }

  // 캐시 초기화
  clearCache(): void {
    this.etagCache.clear();
  }
}

// API Client 인스턴스 생성
export const apiClient = new ApiClient('http://localhost:8080');
```

#### useHotel.ts (React Hook)

```typescript
// useHotel.ts
import { useState, useCallback } from 'react';
import { apiClient, ApiResponse, Hotel, HotelListData, HotelCreateRequest } from './api';

interface UseHotelResult {
  hotel: Hotel | null;
  hotels: Hotel[];
  pagination: any;
  loading: boolean;
  error: string | null;
  getHotel: (id: string) => Promise<void>;
  getHotels: (page?: number, pageSize?: number) => Promise<void>;
  createHotel: (data: HotelCreateRequest) => Promise<boolean>;
  deleteHotel: (id: string) => Promise<boolean>;
}

export const useHotel = (): UseHotelResult => {
  const [hotel, setHotel] = useState<Hotel | null>(null);
  const [hotels, setHotels] = useState<Hotel[]>([]);
  const [pagination, setPagination] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 단일 호텔 조회
  const getHotel = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const response: ApiResponse<Hotel> = await apiClient.get(`/api/v1/hotels/${id}`);

      if (response.status === 'success') {
        setHotel(response.data || null);
      } else if (response.status === 'fail') {
        setError(response.message || '호텔을 찾을 수 없습니다');
      } else {
        setError(response.message || '서버 오류가 발생했습니다');
      }
    } catch (err: any) {
      setError(err.message || '네트워크 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  }, []);

  // 호텔 목록 조회
  const getHotels = useCallback(async (page: number = 1, pageSize: number = 20) => {
    setLoading(true);
    setError(null);

    try {
      const response: ApiResponse<HotelListData> = await apiClient.get(
        `/api/v1/hotels?page=${page}&pageSize=${pageSize}`
      );

      if (response.status === 'success' && response.data) {
        setHotels(response.data.items);
        setPagination(response.data.pagination);
      } else {
        setError(response.message || '호텔 목록을 불러올 수 없습니다');
      }
    } catch (err: any) {
      setError(err.message || '네트워크 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  }, []);

  // 호텔 생성
  const createHotel = useCallback(async (data: HotelCreateRequest): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const response: ApiResponse<Hotel> = await apiClient.post('/api/v1/hotels', data);

      if (response.status === 'success') {
        return true;
      } else if (response.status === 'fail') {
        // 유효성 검증 오류 처리
        const errors = (response.data as any)?.errors || [];
        const errorMessages = errors.map((err: any) => err.message).join('\n');
        setError(errorMessages || response.message || '입력 데이터를 확인해주세요');
        return false;
      } else {
        setError(response.message || '서버 오류가 발생했습니다');
        return false;
      }
    } catch (err: any) {
      setError(err.message || '네트워크 오류가 발생했습니다');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // 호텔 삭제
  const deleteHotel = useCallback(async (id: string): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.delete(`/api/v1/hotels/${id}`);

      if (response.status === 'success') {
        return true;
      } else if (response.status === 'fail') {
        setError(response.message || '호텔을 삭제할 수 없습니다');
        return false;
      } else {
        setError(response.message || '서버 오류가 발생했습니다');
        return false;
      }
    } catch (err: any) {
      setError(err.message || '네트워크 오류가 발생했습니다');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    hotel,
    hotels,
    pagination,
    loading,
    error,
    getHotel,
    getHotels,
    createHotel,
    deleteHotel,
  };
};
```

#### HotelList.tsx (React Component)

```typescript
// HotelList.tsx
import React, { useEffect, useState } from 'react';
import { useHotel } from './useHotel';
import { HotelCreateRequest } from './api';

const HotelList: React.FC = () => {
  const { hotels, pagination, loading, error, getHotels, createHotel, deleteHotel } = useHotel();
  const [currentPage, setCurrentPage] = useState(1);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState<HotelCreateRequest>({
    name: '',
    address: '',
    rating: 0,
  });

  // 컴포넌트 마운트 시 호텔 목록 로드
  useEffect(() => {
    getHotels(currentPage);
  }, [currentPage, getHotels]);

  // 호텔 생성 핸들러
  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    const success = await createHotel(formData);
    if (success) {
      alert('호텔이 생성되었습니다!');
      setShowCreateForm(false);
      setFormData({ name: '', address: '', rating: 0 });
      getHotels(currentPage); // 목록 새로고침
    }
  };

  // 호텔 삭제 핸들러
  const handleDelete = async (id: string) => {
    if (!window.confirm('정말 삭제하시겠습니까?')) {
      return;
    }
    const success = await deleteHotel(id);
    if (success) {
      alert('호텔이 삭제되었습니다!');
      getHotels(currentPage); // 목록 새로고침
    }
  };

  // 페이지 변경 핸들러
  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
  };

  if (loading) {
    return <div className="loading">로딩 중...</div>;
  }

  return (
    <div className="hotel-list-container">
      <h1>호텔 목록</h1>

      {error && <div className="error-message">{error}</div>}

      {/* 호텔 생성 버튼 */}
      <button onClick={() => setShowCreateForm(!showCreateForm)}>
        {showCreateForm ? '취소' : '호텔 추가'}
      </button>

      {/* 호텔 생성 폼 */}
      {showCreateForm && (
        <form onSubmit={handleCreate} className="create-form">
          <div>
            <label>호텔 이름:</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>
          <div>
            <label>주소:</label>
            <input
              type="text"
              value={formData.address}
              onChange={(e) => setFormData({ ...formData, address: e.target.value })}
              required
            />
          </div>
          <div>
            <label>평점 (0-5):</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="5"
              value={formData.rating}
              onChange={(e) => setFormData({ ...formData, rating: parseFloat(e.target.value) })}
              required
            />
          </div>
          <button type="submit">생성</button>
        </form>
      )}

      {/* 호텔 목록 */}
      <div className="hotel-list">
        {hotels.length === 0 ? (
          <p>호텔이 없습니다.</p>
        ) : (
          hotels.map((hotel) => (
            <div key={hotel.id} className="hotel-item">
              <h3>{hotel.name}</h3>
              <p>주소: {hotel.address}</p>
              <p>평점: {hotel.rating} ⭐</p>
              <button onClick={() => handleDelete(hotel.id)}>삭제</button>
            </div>
          ))
        )}
      </div>

      {/* 페이지네이션 */}
      {pagination && (
        <div className="pagination">
          <button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={!pagination.hasPrev}
          >
            이전
          </button>
          <span>
            {currentPage} / {pagination.totalPages} 페이지
          </span>
          <button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={!pagination.hasNext}
          >
            다음
          </button>
          <p>전체 {pagination.totalItems}개</p>
        </div>
      )}
    </div>
  );
};

export default HotelList;
```

----

## 핵심 요약

### JSend 기본 규칙

1. **success**: 성공 시 사용 (2xx)
2. **fail**: 클라이언트 오류 시 사용 (4xx)
3. **error**: 서버 오류 시 사용 (5xx)

### 권장 구조

```typescript
{
  status: "success | fail | error",
  data: { /* 실제 데이터 */ },
  meta: {
    timestamp: "...",
    etag: "...",
    pagination: { /* 페이지 정보 */ }
  },
  message: "...",  // fail, error 시
  code: "..."      // error 시
}
```

### 왜 이 방식이 좋은가?

✅ **단순**: JSend의 3가지 상태만 기억
✅ **명확**: status만 보면 성공/실패 즉시 판단
✅ **실용적**: Google의 유용한 부분 추가
✅ **확장 가능**: meta 섹션으로 향후 확장 용이
✅ **성능**: ETag로 불필요한 데이터 전송 최소화

----

## 참고 자료

- [JSend 공식 스펙](https://github.com/omniti-labs/jsend)
- [Google JSON Style Guide](https://google.github.io/styleguide/jsoncstyleguide.xml)
- [REST API Best Practices](https://restfulapi.net/)
