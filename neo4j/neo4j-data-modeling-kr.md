# Neo4j 데이터 모델링

- [관계형 DB 에서 그래프로](#관계형-db-에서-그래프로)
  - [테이블 → 노드](#테이블--노드)
  - [외래키 → 관계](#외래키--관계)
  - [중간 테이블 → 관계 프로퍼티](#중간-테이블--관계-프로퍼티)
- [모델링 원칙](#모델링-원칙)
  - [노드 vs 프로퍼티 결정 기준](#노드-vs-프로퍼티-결정-기준)
  - [관계 타입 설계](#관계-타입-설계)
- [실전 예제: 소셜 네트워크](#실전-예제-소셜-네트워크)
  - [관계형 DB 설계](#관계형-db-설계)
  - [그래프 설계](#그래프-설계)
  - [쿼리 비교](#쿼리-비교)
- [실전 예제: 영화 추천](#실전-예제-영화-추천)
- [흔한 실수](#흔한-실수)

---

# 관계형 DB 에서 그래프로

## 테이블 → 노드

관계형 DB 의 테이블 하나가 Neo4j 의 레이블 하나에 대응한다.

```
관계형 DB                         Neo4j
─────────                        ──────
users 테이블                      (:User) 노드
  user_id, name, age               {id, name, age}

companies 테이블                  (:Company) 노드
  company_id, name                  {id, name}
```

## 외래키 → 관계

외래키 대신 **직접 관계**로 연결한다.

```
관계형 DB:
  employees 테이블: user_id, company_id (FK)

Neo4j:
  (user:User)-[:WORKS_AT]->(company:Company)
```

예시:
```cypher
// 관계형 DB 에서 하던 JOIN 이 필요 없다
CREATE (alice:User {name: "Alice"})
CREATE (tech:Company {name: "TechCorp"})
CREATE (alice)-[:WORKS_AT {since: 2021}]->(tech)
```

## 중간 테이블 → 관계 프로퍼티

다대다 관계에서 관계형 DB 는 중간 테이블이 필요하다. Neo4j 는 필요 없다.

```
관계형 DB:
  students 테이블
  courses 테이블
  enrollments 테이블 (student_id, course_id, grade, enrolled_at)  ← 중간 테이블

Neo4j:
  (student:Student)-[:ENROLLED_IN {grade: "A", enrolled_at: "2024-03-01"}]->(course:Course)
```

중간 테이블의 컬럼들은 **관계의 프로퍼티**가 된다.

# 모델링 원칙

## 노드 vs 프로퍼티 결정 기준

**"이 값으로 검색하거나 연결할 일이 있는가?"** 가 판단 기준이다.

| 데이터 | 노드로 만들기 | 프로퍼티로 남기기 |
|--------|-------------|----------------|
| 회사 | "같은 회사 사용자 찾기" 필요 → (:Company) 노드 | 단순 표시용이면 `user.company` 프로퍼티 |
| 도시 | "같은 도시 사용자 찾기" 필요 → (:City) 노드 | 단순 표시용이면 `user.city` 프로퍼티 |
| 나이 | 나이로 연결할 일 없음 → `user.age` 프로퍼티 | - |
| 기술 | "같은 기술 가진 사람 찾기" → (:Skill) 노드 | 목록으로만 보여주면 `user.skills` 배열 |

**간단한 규칙**: 처음에는 프로퍼티로 시작하고, 연결이 필요해지면 노드로 승격한다.

## 관계 타입 설계

관계 타입은 **동사형**으로 짓는다:

```
좋은 예:                          나쁜 예:
(:User)-[:FRIEND]->(:User)       (:User)-[:USER_USER]->(:User)
(:User)-[:WORKS_AT]->(:Company)  (:User)-[:COMPANY]->(:Company)
(:User)-[:ATTENDED]->(:School)   (:User)-[:EDUCATION]->(:School)
```

관계 타입이 구체적일수록 쿼리가 명확해지고 성능도 좋아진다:

```cypher
// 구체적인 관계 타입 → 빠른 탐색
MATCH (u)-[:WORKS_AT]->(c:Company) RETURN c

// 포괄적인 관계 타입 → WHERE 로 필터 필요 → 느림
MATCH (u)-[r:RELATED_TO]->(c) WHERE r.type = "works_at" RETURN c
```

# 실전 예제: 소셜 네트워크

## 관계형 DB 설계

```sql
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    company VARCHAR(100),
    school VARCHAR(100)
);

CREATE TABLE friendships (
    user_id_1 INT REFERENCES users(user_id),
    user_id_2 INT REFERENCES users(user_id),
    since DATE,
    PRIMARY KEY (user_id_1, user_id_2)
);
```

## 그래프 설계

```cypher
// 노드
(:User {id, name, company, school})

// 관계
(:User)-[:CONNECTED_TO {since}]->(:User)
```

중간 테이블 `friendships` 가 사라지고, `CONNECTED_TO` 관계로 대체되었다.

## 쿼리 비교

**"민수와 공통 친구가 2명 이상인 비친구 찾기"**

SQL:
```sql
SELECT f2.user_id_2 AS candidate, COUNT(*) AS mutual_count
FROM friendships f1
JOIN friendships f2 ON f1.user_id_2 = f2.user_id_1
WHERE f1.user_id_1 = 1
  AND f2.user_id_2 != 1
  AND f2.user_id_2 NOT IN (SELECT user_id_2 FROM friendships WHERE user_id_1 = 1)
GROUP BY f2.user_id_2
HAVING COUNT(*) >= 2
ORDER BY mutual_count DESC;
```

Cypher:
```cypher
MATCH (u:User {id: 1})-[:CONNECTED_TO]-(f)-[:CONNECTED_TO]-(c)
WHERE u <> c AND NOT (u)-[:CONNECTED_TO]-(c)
WITH c, COUNT(DISTINCT f) AS mutual
WHERE mutual >= 2
ORDER BY mutual DESC
RETURN c.name, mutual
```

Cypher 가 더 **읽기 쉽고**, 데이터가 커질수록 **실행도 빠르다**.

# 실전 예제: 영화 추천

```cypher
// 데이터 모델
CREATE (alice:User {name: "Alice"})
CREATE (bob:User {name: "Bob"})
CREATE (inception:Movie {title: "인셉션", year: 2010})
CREATE (matrix:Movie {title: "매트릭스", year: 1999})
CREATE (interstellar:Movie {title: "인터스텔라", year: 2014})

CREATE (alice)-[:RATED {score: 5}]->(inception)
CREATE (alice)-[:RATED {score: 4}]->(matrix)
CREATE (bob)-[:RATED {score: 5}]->(inception)
CREATE (bob)-[:RATED {score: 5}]->(interstellar)
```

```cypher
// Alice 가 안 본 영화 중, 같은 영화에 높은 점수를 준 사람이 본 영화 추천
MATCH (alice:User {name: "Alice"})-[:RATED]->(m:Movie)<-[:RATED]-(other)
MATCH (other)-[:RATED {score: 5}]->(rec:Movie)
WHERE NOT (alice)-[:RATED]->(rec)
RETURN rec.title AS 추천영화, COUNT(DISTINCT other) AS 추천인수
ORDER BY 추천인수 DESC
```

결과: "인터스텔라" (Bob 이 인셉션을 함께 좋아했고, 인터스텔라에 5점)

# 흔한 실수

| 실수 | 문제 | 해결 |
|------|------|------|
| 모든 것을 노드로 만듦 | 불필요한 복잡도 | 검색/연결 안 하면 프로퍼티 |
| 관계 타입을 하나만 사용 (`RELATED_TO`) | WHERE 필터 필요, 느림 | 구체적 타입 사용 |
| 방향을 무시하고 양방향 모두 생성 | 데이터 중복 | 한쪽만 생성, 쿼리에서 방향 무시 |
| 인덱스 안 만듦 | 노드 검색 느림 | 자주 검색하는 프로퍼티에 인덱스 |
| 노드에 배열 프로퍼티로 관계 저장 | 그래프 탐색 불가 | 관계로 분리 |
