# Neo4j 빠른 시작 (Hands-on)

- [Docker Compose 로 실행하기](#docker-compose-로-실행하기)
  - [docker-compose.yml](#docker-composeyml)
  - [실행 및 확인](#실행-및-확인)
  - [종료](#종료)
- [Neo4j Browser 접속](#neo4j-browser-접속)
- [첫 번째 Cypher 쿼리](#첫-번째-cypher-쿼리)
  - [노드 생성](#노드-생성)
  - [관계 생성](#관계-생성)
  - [데이터 조회](#데이터-조회)
  - [전체 그래프 시각화](#전체-그래프-시각화)
- [소셜 네트워크 예제](#소셜-네트워크-예제)
  - [데이터 입력](#데이터-입력)
  - [친구 찾기](#친구-찾기)
  - [친구의 친구 (FoF) 찾기](#친구의-친구-fof-찾기)
  - [최단 경로 찾기](#최단-경로-찾기)
  - [공통 친구 찾기](#공통-친구-찾기)
- [데이터 초기화](#데이터-초기화)

---

# Docker Compose 로 실행하기

Docker 만 있으면 된다. 별도 설치 불필요.

## docker-compose.yml

아래 내용을 `docker-compose.yml` 파일로 저장한다:

```yaml
services:
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/test1234
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"   # Browser (웹 UI)
      - "7687:7687"   # Bolt (드라이버 연결)
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

- `NEO4J_AUTH`: 사용자명/비밀번호 설정. `neo4j/none` 으로 하면 비밀번호 없이 접속.
- `7474`: 웹 브라우저 UI.
- `7687`: Bolt 프로토콜 (Python, Java 등 드라이버가 사용하는 포트).
- `APOC`: Neo4j 의 확장 라이브러리 (유틸리티 함수 모음).

## 실행 및 확인

```bash
docker-compose up -d

# 로그 확인 (Started. 가 보이면 준비 완료)
docker-compose logs -f neo4j
```

약 10-20초 후 `Started.` 메시지가 출력된다.

## 종료

```bash
docker-compose down       # 컨테이너 종료 (데이터 유지)
docker-compose down -v    # 컨테이너 + 데이터 삭제 (초기화)
```

# Neo4j Browser 접속

브라우저에서 **http://localhost:7474** 을 연다.

- Connect URL: `neo4j://localhost:7687`
- Username: `neo4j`
- Password: `test1234`

접속하면 Cypher 쿼리를 직접 입력하고 결과를 **그래프로 시각화**할 수 있다.

# 첫 번째 Cypher 쿼리

Neo4j Browser 의 상단 입력창에 쿼리를 입력하고 실행(▶ 또는 Ctrl+Enter) 한다.

## 노드 생성

```cypher
// 사람 노드 3개 생성
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 28})
CREATE (charlie:Person {name: "Charlie", age: 35})
```

실행 후 "Created 3 nodes" 메시지가 나온다.

## 관계 생성

```cypher
// Alice 와 Bob 을 친구로 연결
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:FRIEND {since: 2020}]->(b)
```

```cypher
// Bob 과 Charlie 를 친구로 연결
MATCH (b:Person {name: "Bob"}), (c:Person {name: "Charlie"})
CREATE (b)-[:FRIEND {since: 2022}]->(c)
```

## 데이터 조회

```cypher
// Alice 의 친구 찾기
MATCH (a:Person {name: "Alice"})-[:FRIEND]->(friend)
RETURN friend.name AS name, friend.age AS age
```

결과:
```
╒══════╤═════╕
│name  │age  │
╞══════╪═════╡
│"Bob" │28   │
╘══════╧═════╛
```

## 전체 그래프 시각화

```cypher
// 모든 노드와 관계를 한눈에 보기
MATCH (n) RETURN n
```

Browser 가 노드를 **원**, 관계를 **화살표**로 시각화해서 보여준다.

# 소셜 네트워크 예제

좀 더 현실적인 예제를 만들어보자. 먼저 기존 데이터를 지운다:

```cypher
MATCH (n) DETACH DELETE n
```

## 데이터 입력

```cypher
// 사용자 6명 생성
CREATE (a:User {name: "민수", company: "테크코프", school: "서울대"})
CREATE (b:User {name: "영희", company: "테크코프", school: "KAIST"})
CREATE (c:User {name: "지훈", company: "데이터인크", school: "서울대"})
CREATE (d:User {name: "수정", company: "데이터인크", school: "연세대"})
CREATE (e:User {name: "대현", company: "AI랩", school: "KAIST"})
CREATE (f:User {name: "서윤", company: "AI랩", school: "서울대"})

// 연결 관계 생성
CREATE (a)-[:CONNECTED_TO]->(b)    // 민수 - 영희
CREATE (a)-[:CONNECTED_TO]->(c)    // 민수 - 지훈
CREATE (b)-[:CONNECTED_TO]->(d)    // 영희 - 수정
CREATE (b)-[:CONNECTED_TO]->(e)    // 영희 - 대현
CREATE (c)-[:CONNECTED_TO]->(d)    // 지훈 - 수정
CREATE (d)-[:CONNECTED_TO]->(f)    // 수정 - 서윤
CREATE (e)-[:CONNECTED_TO]->(f)    // 대현 - 서윤
```

그래프 구조:

```
민수 --- 영희 --- 수정 --- 서윤
  \              /        /
   --- 지훈 ---    대현 --
        영희 --- 대현
```

## 친구 찾기

```cypher
// 민수의 직접 연결 (1-hop)
MATCH (u:User {name: "민수"})-[:CONNECTED_TO]-(friend)
RETURN friend.name AS 이름, friend.company AS 회사
```

결과:
```
╒══════╤═══════════╕
│이름   │회사        │
╞══════╪═══════════╡
│"영희" │"테크코프"   │
│"지훈" │"데이터인크" │
╘══════╧═══════════╛
```

방향 없이 `-[:CONNECTED_TO]-` 으로 쓰면 양방향 모두 탐색한다.

## 친구의 친구 (FoF) 찾기

```cypher
// 민수의 FoF (2-hop) — 이미 친구인 사람 제외
MATCH (u:User {name: "민수"})-[:CONNECTED_TO]-(f)-[:CONNECTED_TO]-(fof)
WHERE u <> fof AND NOT (u)-[:CONNECTED_TO]-(fof)
WITH fof, COUNT(DISTINCT f) AS 공통친구수
ORDER BY 공통친구수 DESC
RETURN fof.name AS 이름, fof.company AS 회사, 공통친구수
```

결과:
```
╒══════╤═══════════╤══════════╕
│이름   │회사        │공통친구수  │
╞══════╪═══════════╪══════════╡
│"수정" │"데이터인크" │2         │  ← 영희, 지훈 둘 다 통해 연결
│"대현" │"AI랩"     │1         │  ← 영희 통해 연결
╘══════╧═══════════╧══════════╛
```

수정이 공통 친구 2명으로 1위 — "알 수도 있는 사람" 추천의 핵심 로직이다.

## 최단 경로 찾기

```cypher
// 민수에서 서윤까지 최단 경로
MATCH path = shortestPath(
  (a:User {name: "민수"})-[:CONNECTED_TO*]-(b:User {name: "서윤"})
)
RETURN [n IN nodes(path) | n.name] AS 경로, length(path) AS 거리
```

결과:
```
╒══════════════════════════════════╤══════╕
│경로                               │거리   │
╞══════════════════════════════════╪══════╡
│["민수", "영희", "수정", "서윤"]     │3     │
╘══════════════════════════════════╧══════╛
```

이런 쿼리는 SQL 로는 매우 복잡하지만, Cypher 로는 한 줄이다.

## 공통 친구 찾기

```cypher
// 민수와 수정의 공통 친구
MATCH (a:User {name: "민수"})-[:CONNECTED_TO]-(mutual)-[:CONNECTED_TO]-(b:User {name: "수정"})
RETURN mutual.name AS 공통친구
```

결과:
```
╒═════════╕
│공통친구   │
╞═════════╡
│"영희"    │
│"지훈"    │
╘═════════╛
```

# 데이터 초기화

실습이 끝나면 모든 데이터를 지울 수 있다:

```cypher
// 모든 노드와 관계 삭제
MATCH (n) DETACH DELETE n
```

또는 Docker 볼륨을 삭제:

```bash
docker-compose down -v
```
