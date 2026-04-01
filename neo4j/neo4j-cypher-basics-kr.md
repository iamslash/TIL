# Cypher 기초

- [Cypher 란](#cypher-란)
- [CRUD 기본](#crud-기본)
  - [CREATE: 생성](#create-생성)
  - [MATCH: 조회](#match-조회)
  - [SET: 수정](#set-수정)
  - [DELETE: 삭제](#delete-삭제)
  - [MERGE: 있으면 조회, 없으면 생성](#merge-있으면-조회-없으면-생성)
- [패턴 매칭](#패턴-매칭)
  - [관계 방향](#관계-방향)
  - [가변 길이 경로](#가변-길이-경로)
  - [필터링 (WHERE)](#필터링-where)
- [집계와 정렬](#집계와-정렬)
  - [COUNT, SUM, AVG](#count-sum-avg)
  - [ORDER BY, LIMIT](#order-by-limit)
  - [WITH: 파이프라인](#with-파이프라인)
- [실전 패턴](#실전-패턴)
  - [FoF (친구의 친구)](#fof-친구의-친구)
  - [최단 경로](#최단-경로)
  - [추천: 같은 회사 + 공통 친구](#추천-같은-회사--공통-친구)
- [인덱스와 제약조건](#인덱스와-제약조건)

---

# Cypher 란

Cypher 는 Neo4j 의 **쿼리 언어**이다. SQL 이 테이블을 다루듯, Cypher 는
그래프를 다룬다.

핵심 문법은 **ASCII 아트**로 패턴을 표현하는 것이다:

```
(node)            -- 노드: 괄호
-[relationship]-> -- 관계: 대괄호 + 화살표
```

예시:
```cypher
(alice)-[:FRIEND]->(bob)     -- Alice 가 Bob 의 친구
(a:Person)-[:WORKS_AT]->(c:Company)  -- Person 이 Company 에서 일함
```

SQL 과 비교:

| SQL | Cypher |
|-----|--------|
| `SELECT` | `RETURN` |
| `FROM + JOIN` | `MATCH (패턴)` |
| `WHERE` | `WHERE` |
| `INSERT INTO` | `CREATE` |
| `UPDATE` | `SET` |
| `DELETE FROM` | `DELETE` / `DETACH DELETE` |

# CRUD 기본

## CREATE: 생성

```cypher
// 노드 생성
CREATE (p:Person {name: "Alice", age: 30})
RETURN p

// 관계와 함께 생성
CREATE (a:Person {name: "Alice"})-[:FRIEND]->(b:Person {name: "Bob"})
```

## MATCH: 조회

```cypher
// 모든 Person 노드 조회
MATCH (p:Person) RETURN p

// 이름으로 찾기
MATCH (p:Person {name: "Alice"}) RETURN p

// 관계 패턴으로 찾기
MATCH (a:Person)-[:FRIEND]->(b:Person)
RETURN a.name AS 친구1, b.name AS 친구2
```

## SET: 수정

```cypher
// 프로퍼티 추가/변경
MATCH (p:Person {name: "Alice"})
SET p.age = 31, p.city = "서울"
RETURN p
```

## DELETE: 삭제

```cypher
// 관계 삭제
MATCH (a:Person {name: "Alice"})-[r:FRIEND]->(b:Person {name: "Bob"})
DELETE r

// 노드 삭제 (관계가 없는 노드만 가능)
MATCH (p:Person {name: "Alice"})
DELETE p

// 노드 + 연결된 모든 관계 함께 삭제
MATCH (p:Person {name: "Alice"})
DETACH DELETE p
```

`DETACH DELETE` 는 노드에 연결된 관계를 먼저 지우고 노드를 지운다.
관계가 있는 노드를 그냥 `DELETE` 하면 에러가 발생한다.

## MERGE: 있으면 조회, 없으면 생성

```cypher
// Alice 가 없으면 생성, 있으면 기존 노드 사용
MERGE (p:Person {name: "Alice"})
ON CREATE SET p.age = 30
ON MATCH SET p.lastSeen = datetime()
RETURN p
```

`MERGE` 는 데이터 동기화에 매우 유용하다. 외부 데이터를 반복적으로 넣을 때
중복 생성을 방지한다.

# 패턴 매칭

## 관계 방향

```cypher
// 나가는 방향
MATCH (a)-[:FRIEND]->(b) RETURN a, b

// 들어오는 방향
MATCH (a)<-[:FRIEND]-(b) RETURN a, b

// 방향 무시 (양방향 탐색)
MATCH (a)-[:FRIEND]-(b) RETURN a, b
```

소셜 네트워크에서 "친구" 는 보통 양방향이므로 방향 없이 `-[:FRIEND]-` 로 쓰는
경우가 많다.

## 가변 길이 경로

```cypher
// 정확히 2-hop
MATCH (a)-[:FRIEND*2]-(b) RETURN b

// 1~3 hop 범위
MATCH (a)-[:FRIEND*1..3]-(b) RETURN b

// 최소 2-hop, 최대 제한 없음 (주의: 성능)
MATCH (a)-[:FRIEND*2..]-(b) RETURN b
```

`*2..3` 은 "2번 이상 3번 이하 관계를 따라가라" 는 뜻이다.

## 필터링 (WHERE)

```cypher
// 나이가 25 이상인 친구
MATCH (a:Person)-[:FRIEND]-(b:Person)
WHERE a.name = "Alice" AND b.age >= 25
RETURN b

// 특정 관계가 없는 노드 (NOT EXISTS)
MATCH (a:Person), (b:Person)
WHERE a.name = "Alice" AND NOT (a)-[:FRIEND]-(b)
RETURN b.name AS 친구_아닌_사람

// 프로퍼티 존재 여부
MATCH (p:Person)
WHERE p.email IS NOT NULL
RETURN p
```

# 집계와 정렬

## COUNT, SUM, AVG

```cypher
// 각 사용자의 친구 수
MATCH (u:User)-[:CONNECTED_TO]-(friend)
RETURN u.name AS 이름, COUNT(friend) AS 친구수
```

```cypher
// 회사별 직원 수
MATCH (u:User)
WHERE u.company IS NOT NULL
RETURN u.company AS 회사, COUNT(u) AS 직원수
ORDER BY 직원수 DESC
```

## ORDER BY, LIMIT

```cypher
// 친구 많은 순으로 상위 5명
MATCH (u:User)-[:CONNECTED_TO]-(friend)
RETURN u.name, COUNT(friend) AS 친구수
ORDER BY 친구수 DESC
LIMIT 5
```

## WITH: 파이프라인

`WITH` 는 Cypher 의 핵심 기능이다. SQL 의 서브쿼리와 비슷하지만, **데이터를
파이프라인처럼 다음 단계로 전달**한다.

```cypher
// 1단계: 친구 수 계산 → 2단계: 10명 이상만 필터
MATCH (u:User)-[:CONNECTED_TO]-(friend)
WITH u, COUNT(friend) AS 친구수
WHERE 친구수 >= 10
RETURN u.name, 친구수
ORDER BY 친구수 DESC
```

# 실전 패턴

## FoF (친구의 친구)

```cypher
// 사용자 1 의 FoF, 공통 친구 수 포함
MATCH (u:User {id: 1})-[:CONNECTED_TO]-(f)-[:CONNECTED_TO]-(fof)
WHERE u <> fof AND NOT (u)-[:CONNECTED_TO]-(fof)
WITH fof, COUNT(DISTINCT f) AS mutual
ORDER BY mutual DESC
LIMIT 20
RETURN fof.name, fof.company, mutual AS 공통친구수
```

## 최단 경로

```cypher
// 두 사용자 사이의 최단 경로
MATCH path = shortestPath(
  (a:User {name: "민수"})-[:CONNECTED_TO*]-(b:User {name: "서윤"})
)
RETURN [n IN nodes(path) | n.name] AS 경로, length(path) AS 거리
```

```cypher
// 모든 최단 경로 (여러 개일 수 있음)
MATCH path = allShortestPaths(
  (a:User {name: "민수"})-[:CONNECTED_TO*]-(b:User {name: "서윤"})
)
RETURN [n IN nodes(path) | n.name] AS 경로
```

## 추천: 같은 회사 + 공통 친구

```cypher
// 같은 회사에서 일하면서 공통 친구가 있는 사람
MATCH (u:User {name: "민수"})-[:CONNECTED_TO]-(mutual)-[:CONNECTED_TO]-(rec)
WHERE u <> rec
  AND NOT (u)-[:CONNECTED_TO]-(rec)
  AND rec.company = u.company
WITH rec, COUNT(DISTINCT mutual) AS 공통수
ORDER BY 공통수 DESC
RETURN rec.name, rec.company, 공통수
```

이 쿼리를 SQL 로 작성하면 3개 이상의 JOIN 이 필요하지만, Cypher 는 패턴
그대로 읽힌다.

# 인덱스와 제약조건

```cypher
// 인덱스 생성 (조회 성능 향상)
CREATE INDEX FOR (u:User) ON (u.id)
CREATE INDEX FOR (u:User) ON (u.name)
CREATE INDEX FOR (u:User) ON (u.company)

// 유니크 제약조건 (중복 방지)
CREATE CONSTRAINT FOR (u:User) REQUIRE u.id IS UNIQUE

// 인덱스 목록 확인
SHOW INDEXES
```

인덱스는 **노드를 처음 찾을 때** 사용된다 (예: `MATCH (u:User {id: 1})`).
이후 관계를 따라가는 탐색은 인덱스 없이 포인터로 직접 이동한다.
