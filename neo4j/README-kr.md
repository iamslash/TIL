# Neo4j 한글 가이드

- [개요](#개요)
- [왜 그래프 데이터베이스인가](#왜-그래프-데이터베이스인가)
  - [관계형 DB 의 한계](#관계형-db-의-한계)
  - [그래프 DB 가 빛나는 순간](#그래프-db-가-빛나는-순간)
- [핵심 개념](#핵심-개념)
  - [노드 (Node)](#노드-node)
  - [관계 (Relationship)](#관계-relationship)
  - [레이블 (Label)](#레이블-label)
  - [프로퍼티 (Property)](#프로퍼티-property)
  - [그래프 데이터 모델 예시](#그래프-데이터-모델-예시)
- [아키텍처](#아키텍처)
  - [저장 엔진](#저장-엔진)
  - [인덱스](#인덱스)
  - [트랜잭션](#트랜잭션)
  - [Community vs Enterprise](#community-vs-enterprise)
- [학습 자료](#학습-자료)
- [관련 문서](#관련-문서)

---

# 개요

Neo4j 는 **그래프 데이터베이스**이다. 데이터를 테이블이 아니라 **노드(점)와
관계(선)**로 저장한다.

쉽게 말해, 소셜 네트워크의 친구 관계, 추천 시스템, 지식 그래프 같은 **"연결"이
핵심인 데이터**를 다룰 때 관계형 DB 보다 훨씬 자연스럽고 빠른 데이터베이스라고
생각하면 된다.

```
관계형 DB:  테이블 + JOIN → 연결이 많을수록 느려짐
Neo4j:     노드 + 관계 → 연결 탐색이 본질적으로 빠름
```

# 왜 그래프 데이터베이스인가

## 관계형 DB 의 한계

"사용자 A 의 친구의 친구를 찾아라" 를 SQL 로 작성하면:

```sql
-- 2-hop: 친구의 친구
SELECT DISTINCT e2.user_id_2 AS fof
FROM edges e1
JOIN edges e2 ON e1.user_id_2 = e2.user_id_1
WHERE e1.user_id_1 = 1
  AND e2.user_id_2 != 1
  AND e2.user_id_2 NOT IN (SELECT user_id_2 FROM edges WHERE user_id_1 = 1);
```

3-hop 이 되면:

```sql
-- 3-hop: 친구의 친구의 친구 → JOIN 이 3개
SELECT DISTINCT e3.user_id_2 AS fofof
FROM edges e1
JOIN edges e2 ON e1.user_id_2 = e2.user_id_1
JOIN edges e3 ON e2.user_id_2 = e3.user_id_1
WHERE e1.user_id_1 = 1
  AND e3.user_id_2 != 1
  AND ...
```

**문제점**:
- hop 이 늘어날 때마다 JOIN 이 추가된다.
- 각 JOIN 은 테이블 전체를 스캔할 수 있다.
- 100만 행 × 100만 행 JOIN → 매우 느림.

## 그래프 DB 가 빛나는 순간

같은 질문을 Neo4j 의 Cypher 로 작성하면:

```cypher
-- 2-hop
MATCH (u:User {id: 1})-[:FRIEND]-()-[:FRIEND]-(fof)
WHERE NOT (u)-[:FRIEND]-(fof)
RETURN fof

-- 3-hop (한 줄만 바꾸면 됨)
MATCH (u:User {id: 1})-[:FRIEND*2..3]-(fof)
WHERE NOT (u)-[:FRIEND]-(fof)
RETURN fof
```

**왜 빠른가**:
- Neo4j 는 각 노드에서 연결된 관계를 **포인터로 직접 따라간다** (index-free adjacency).
- JOIN 이 아니라 **포인터 점프**이므로 데이터가 아무리 많아도 탐색 시간이 관계 수에만 비례한다.
- 100만 명 중 친구 1,000명의 FoF 를 찾는 데 걸리는 시간: 관계형 DB 는 수 초, Neo4j 는 수십 ms.

```
               관계형 DB              Neo4j
2-hop          빠름                   빠름
3-hop          느려짐 (3중 JOIN)       빠름 (포인터 3번 점프)
5-hop          사실상 불가             여전히 가능
최단 경로       직접 BFS 구현 필요       내장 함수 한 줄
커뮤니티 감지    불가                   GDS 라이브러리
```

# 핵심 개념

## 노드 (Node)

그래프의 **점**. 사람, 영화, 도시 등 하나의 엔티티를 나타낸다.

```cypher
CREATE (alice:Person {name: "Alice", age: 30})
```

`alice` 라는 변수에 `Person` 레이블을 가진 노드를 생성했다.

## 관계 (Relationship)

그래프의 **선**. 두 노드를 연결한다. 항상 **방향**과 **타입**이 있다.

```cypher
CREATE (alice)-[:FRIEND {since: 2020}]->(bob)
```

Alice 에서 Bob 으로 `FRIEND` 관계를 만들었다. `since` 는 관계의 프로퍼티이다.

## 레이블 (Label)

노드의 **종류**를 나타낸다. 하나의 노드에 여러 레이블을 붙일 수 있다.

```cypher
CREATE (alice:Person:Employee {name: "Alice"})
```

Alice 는 `Person` 이면서 동시에 `Employee` 이다.

## 프로퍼티 (Property)

노드나 관계에 붙는 **키-값 쌍**. JSON 과 비슷하다.

```cypher
CREATE (movie:Movie {title: "인셉션", year: 2010, rating: 8.8})
```

## 그래프 데이터 모델 예시

소셜 네트워크를 그래프로 모델링하면:

```
(:Person {name: "Alice"})
    -[:FRIEND {since: 2020}]->
(:Person {name: "Bob"})
    -[:WORKS_AT]->
(:Company {name: "TechCorp"})

(:Person {name: "Alice"})
    -[:ATTENDED]->
(:School {name: "서울대"})
```

관계형 DB 였다면:
- `persons` 테이블
- `companies` 테이블
- `schools` 테이블
- `person_friends` 중간 테이블
- `person_companies` 중간 테이블
- `person_schools` 중간 테이블

Neo4j 에서는 **중간 테이블 없이 직접 연결**한다.

# 아키텍처

## 저장 엔진

Neo4j 는 데이터를 두 개의 저장소에 나눠서 저장한다:

```
노드 저장소:    [노드 ID] → [첫 번째 관계 포인터, 첫 번째 프로퍼티 포인터, 레이블]
관계 저장소:    [관계 ID] → [시작 노드, 끝 노드, 관계 타입, 다음 관계 포인터]
```

핵심은 **다음 관계 포인터**이다. 노드에서 관계를 따라갈 때 인덱스 검색 없이
포인터를 따라가기만 하면 된다. 이것이 "index-free adjacency" 이고, 그래프
탐색이 빠른 근본적인 이유이다.

## 인덱스

포인터 탐색은 "이 노드에서 연결된 관계 따라가기" 에 최적화되어 있다.
**"이름이 Alice 인 노드 찾기"** 같은 조회는 인덱스가 필요하다.

```cypher
-- 인덱스 생성
CREATE INDEX FOR (p:Person) ON (p.name)

-- 이후 이 쿼리가 빨라짐
MATCH (p:Person {name: "Alice"}) RETURN p
```

## 트랜잭션

Neo4j 는 **ACID 트랜잭션**을 완전히 지원한다. 관계형 DB 와 동일한 수준의
데이터 일관성을 보장한다.

## Community vs Enterprise

| 항목 | Community (무료) | Enterprise (유료) |
|------|-----------------|------------------|
| 라이선스 | GPL-3.0 | 상용 |
| 클러스터링 | 단일 서버 | Causal Cluster (HA) |
| 성능 | 프로덕션 가능 | 더 빠른 캐시, 병렬 처리 |
| 학습/개발 | 충분 | 대규모 프로덕션 |
| Docker 이미지 | `neo4j:5-community` | `neo4j:5-enterprise` |

학습과 개발에는 Community Edition 이면 충분하다.

# 학습 자료

- [Neo4j 공식 문서](https://neo4j.com/docs/)
- [Cypher 매뉴얼](https://neo4j.com/docs/cypher-manual/current/)
- [Graph Data Science (GDS)](https://neo4j.com/docs/graph-data-science/current/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)

# 관련 문서

- [빠른 시작 (Docker Compose)](neo4j-quickstart-kr.md) — 설치부터 첫 Cypher 쿼리까지
- [Cypher 기초](neo4j-cypher-basics-kr.md) — 쿼리 언어 문법과 예제
- [데이터 모델링](neo4j-data-modeling-kr.md) — 관계형 DB 에서 그래프로 전환
- [Python 연동](neo4j-python-kr.md) — neo4j 드라이버 사용법
