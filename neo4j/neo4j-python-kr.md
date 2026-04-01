# Neo4j Python 연동

- [설치](#설치)
- [기본 연결](#기본-연결)
- [CRUD 예제](#crud-예제)
  - [노드 생성](#노드-생성)
  - [관계 생성](#관계-생성)
  - [조회](#조회)
  - [수정과 삭제](#수정과-삭제)
- [트랜잭션](#트랜잭션)
- [실전 예제: PYMK 후보 생성](#실전-예제-pymk-후보-생성)
- [Docker Compose 연동](#docker-compose-연동)
- [주의사항](#주의사항)

---

# 설치

```bash
pip install neo4j
```

# 기본 연결

```python
from neo4j import GraphDatabase

# 연결
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "test1234"),
)

# 연결 확인
driver.verify_connectivity()
print("Connected!")

# 사용 후 반드시 닫기
driver.close()
```

`bolt://` 은 Neo4j 의 바이너리 프로토콜이다. HTTP (`http://localhost:7474`)
보다 빠르다.

# CRUD 예제

## 노드 생성

```python
def create_user(driver, name, company):
    with driver.session() as session:
        result = session.run(
            "CREATE (u:User {name: $name, company: $company}) RETURN u",
            name=name, company=company,
        )
        record = result.single()
        print(f"Created: {record['u']}")

create_user(driver, "민수", "테크코프")
create_user(driver, "영희", "테크코프")
create_user(driver, "지훈", "데이터인크")
```

**중요**: Cypher 쿼리에 값을 직접 넣지 말고 반드시 `$파라미터`를 사용한다.
SQL injection 과 동일한 보안 위험이 있다.

```python
# 나쁜 예 (절대 하지 말 것)
session.run(f"CREATE (u:User {{name: '{name}'}})")

# 좋은 예
session.run("CREATE (u:User {name: $name})", name=name)
```

## 관계 생성

```python
def create_friendship(driver, name1, name2):
    with driver.session() as session:
        session.run(
            """
            MATCH (a:User {name: $name1}), (b:User {name: $name2})
            MERGE (a)-[:CONNECTED_TO]->(b)
            MERGE (b)-[:CONNECTED_TO]->(a)
            """,
            name1=name1, name2=name2,
        )
        print(f"Connected: {name1} <-> {name2}")

create_friendship(driver, "민수", "영희")
create_friendship(driver, "민수", "지훈")
```

`MERGE` 를 사용하면 이미 관계가 있을 때 중복 생성하지 않는다.

## 조회

```python
def get_friends(driver, name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (u:User {name: $name})-[:CONNECTED_TO]-(friend)
            RETURN friend.name AS name, friend.company AS company
            """,
            name=name,
        )
        friends = [(r["name"], r["company"]) for r in result]
        print(f"{name}의 친구: {friends}")
        return friends

get_friends(driver, "민수")
# 민수의 친구: [('영희', '테크코프'), ('지훈', '데이터인크')]
```

`result` 는 이터레이터이다. 한 번만 순회할 수 있다.
여러 번 사용하려면 리스트로 변환해둔다.

## 수정과 삭제

```python
# 수정
with driver.session() as session:
    session.run(
        "MATCH (u:User {name: $name}) SET u.age = $age",
        name="민수", age=30,
    )

# 노드 + 관계 삭제
with driver.session() as session:
    session.run(
        "MATCH (u:User {name: $name}) DETACH DELETE u",
        name="지훈",
    )
```

# 트랜잭션

```python
def transfer_data(tx, source, target):
    """트랜잭션 함수: 성공하면 커밋, 실패하면 자동 롤백"""
    tx.run(
        "MATCH (a:User {name: $source})-[r:CONNECTED_TO]->(b) "
        "DELETE r",
        source=source,
    )
    tx.run(
        "MATCH (a:User {name: $source}), (b:User {name: $target}) "
        "CREATE (b)-[:CONNECTED_TO]->(a)",
        source=source, target=target,
    )

# execute_write 로 트랜잭션 실행
with driver.session() as session:
    session.execute_write(transfer_data, "민수", "영희")
```

`execute_write` 는:
- 함수 전체가 성공하면 **자동 커밋**
- 중간에 에러가 나면 **자동 롤백**
- 일시적 에러(네트워크 등)는 **자동 재시도**

# 실전 예제: PYMK 후보 생성

```python
def get_pymk_candidates(driver, user_id, limit=20):
    """FoF 기반 PYMK 후보를 Neo4j 에서 직접 생성"""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (u:User {id: $uid})-[:CONNECTED_TO]-(f)-[:CONNECTED_TO]-(candidate)
            WHERE u <> candidate AND NOT (u)-[:CONNECTED_TO]-(candidate)
            WITH candidate, COUNT(DISTINCT f) AS mutual_friends
            ORDER BY mutual_friends DESC
            LIMIT $limit
            RETURN candidate.id AS id,
                   candidate.name AS name,
                   candidate.company AS company,
                   mutual_friends
            """,
            uid=user_id, limit=limit,
        )
        candidates = []
        for r in result:
            candidates.append({
                "id": r["id"],
                "name": r["name"],
                "company": r["company"],
                "mutual_friends": r["mutual_friends"],
            })
        return candidates

# 사용
candidates = get_pymk_candidates(driver, user_id=1, limit=10)
for c in candidates:
    print(f"  {c['name']} ({c['company']}) - 공통 친구 {c['mutual_friends']}명")
```

# Docker Compose 연동

Python 앱과 Neo4j 를 함께 실행하는 예시:

```yaml
services:
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/test1234
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD-SHELL", "neo4j status || exit 1"]
      interval: 10s
      timeout: 10s
      retries: 10

  app:
    build: .
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: test1234
    depends_on:
      neo4j:
        condition: service_healthy

volumes:
  neo4j_data:
```

Python 코드에서:

```python
import os
from neo4j import GraphDatabase

uri = os.environ["NEO4J_URI"]        # bolt://neo4j:7687
user = os.environ["NEO4J_USER"]      # neo4j
password = os.environ["NEO4J_PASSWORD"]  # test1234

driver = GraphDatabase.driver(uri, auth=(user, password))
```

Docker Compose 네트워크 안에서는 `neo4j` (서비스명)로 접근한다.
호스트에서 직접 접근할 때는 `localhost:7687` 을 사용한다.

# 주의사항

| 주의 | 설명 |
|------|------|
| 드라이버 닫기 | `driver.close()` 를 반드시 호출. 안 하면 연결 누출 |
| 파라미터 사용 | `$name` 형태. f-string 으로 직접 넣지 말 것 |
| 세션은 짧게 | `with driver.session()` 패턴 사용. 오래 열어두지 말 것 |
| 대량 삽입 | 개별 `CREATE` 대신 `UNWIND $rows AS r CREATE ...` 배치 사용 |
| 재시도 | `execute_write` / `execute_read` 사용하면 일시 에러 자동 재시도 |
| 인덱스 | 자주 검색하는 프로퍼티에 인덱스 생성. 없으면 full scan |
