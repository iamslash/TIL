- [Materials](#materials)
- [Abstract](#abstract)
- [Dirty Read](#dirty-read)
- [Non-repeatable Read](#non-repeatable-read)
- [Phantom Read](#phantom-read)
- [Lost Update](#lost-update)
- [Serializable Anomaly](#serializable-anomaly)

----

# Materials

- [SQL 트랜잭션 - 믿는 도끼에 발등 찍힌다](https://blog.sapzil.org/2017/04/01/do-not-trust-sql-transaction/)
  - non-repeatable read 를 설명한다.

# Abstract

Data Base Concurrency Problem 의 종류는 다음과 같다.

- dirty read
- non-repeatable read
- phantom read
- lost update
- serializable anomaly???

# Dirty Read

- A transaction 이 값을 1 에서 2 로 수정하고 아직 commit 하지 않았다. B transaction 은 값을 2 로 읽어들인다. 만약 A transaction 이 rollback 되면 B transaction 은 잘못된 값 2 을 읽게 된다.

```sql
-- Transaction A
BEGIN; -- 트랜잭션 시작
UPDATE accounts SET balance = 2 WHERE id = 1; -- 아직 커밋하지 않음

-- Transaction B
SELECT balance FROM accounts WHERE id = 1; -- 2를 읽음, 하지만 A가 롤백될 경우 잘못된 값

-- Transaction A
-- 롤백되는 경우를 가정
ROLLBACK; -- 변경 사항을 롤백
```

- Rollback 후 잘못된 값을 읽는다.
- Dirty read 문제를 해결하기 위해서는 트랜잭션의 격리 수준(Isolation Level)을 조정할 수 있습니다. 대부분의 데이터베이스 시스템에서는 다음과 같은 격리 수준을 제공합니다: `Read Uncommitted, Read Committed, Repeatable Read, Serializable`. Dirty read를 방지하기 위해서는 최소한 `Read Committed` 격리 수준을 사용해야 합니다.

```sql
-- Read Committed Isolation Level 설정 예시 (PostgreSQL):
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT balance FROM accounts WHERE id = 1; -- A가 커밋하지 않았다면, 변경 전 값을 읽음
COMMIT;
```

# Non-repeatable Read

- A transaction 이 한번 읽어온다. B transaction 이 Update 한다. A transaction 이 다시 한번 읽어온다. 이때 처음 읽었던 값과 다른 값을 읽어온다.
  
    ```sql
    BEGIN TRAN
      SELECT SUM(Revenue) AS Total FROM Data;
      --another trx updates a row
      SELECT Revenue AS Detail FROM Data;
    COMMIT  
    ```

- 다시 읽을 때 잘못된 값을 읽는다.
- Non-repeatable read 문제를 해결하기 위해, 데이터베이스 트랜잭션의 격리 수준(Isolation Level)을 조정하여, 한 트랜잭션에서 조회한 데이터가 다른 트랜잭션에 의해 변경되는 것을 방지할 수 있습니다. 이를 위해 `Repeatable Read` 이상의 격리 수준을 사용하면 됩니다.

```sql
-- Repeatable Read Isolation Level 설정 예시 (PostgreSQL):

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT SUM(Revenue) AS Total FROM Data; -- 첫 번째 조회
-- 다른 트랜잭션의 업데이트는 이 트랜잭션에 영향을 주지 않음
SELECT SUM(Revenue) AS Total FROM Data; -- 두 번째 조회, 첫 번째 조회와 동일한 결과 보장
COMMIT;
```

# Phantom Read

- A transaction 이 한번 읽어온다. B transaction 이 insert 한다. A transaction 이 다시 한번 읽어온다. 이때 처음 읽었던 record 들에 하나 더 추가된 혹은 하나 삭제된 record 들을 읽어온다.

    ```sql
    BEGIN TRAN
      SELECT SUM(Revenue) AS Total FROM Data;
      --another trx inserts/deletes a row
      SELECT Revenue AS Detail FROM Data;
    COMMIT  
    ```
- record 가 도깨비 처럼 나타나던가 없어진다.
- Phantom read 문제를 해결하기 위해, 데이터베이스 트랜잭션의 격리 수준(Isolation Level)을 조정하여, 한 트랜잭션에서 조회한 결과 범위 내에 새로운 데이터가 삽입되거나 삭제되는 것을 방지할 수 있습니다. 이를 위해 `Serializable` 격리 수준을 사용하면 됩니다. 그러나 MySQL 은 isolation level 을 repeatable read 로 하더라도 phantom read 를 해결한다. [Repeatable Read Isolation Level In MySQL](/isolation/README.md#practice-of-repeatable-read)

    ```sql
    -- Serializable Isolation Level 설정 예시 (PostgreSQL):

    SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    BEGIN;
    SELECT SUM(Revenue) AS Total FROM Data; -- 첫 번째 조회
    -- 다른 트랜잭션의 삽입/삭제 작업은 이 트랜잭션에 영향을 주지 않음
    SELECT SUM(Revenue) AS Total FROM Data; -- 두 번째 조회, 첫 번째 조회와 동일한 결과 보장
    COMMIT;
    ```

# Lost Update

- 두 개 이상의 트랜잭션이 동시에 같은 데이터를 수정하려고 할 때 발생하는 동시성 문제입니다. 한 트랜잭션의 변경이 다른 트랜잭션에 의해 덮어쓰여지는 경우를 말합니다. 
  
    ```sql
    --계좌 잔액: 1000

    -- 트랜잭션 A 시작
    START TRANSACTION;
    SELECT balance FROM accounts WHERE id = 1; -- 1000 반환
    -- 잔액을 1200으로 업데이트하려고 계산 중 (예: 200 추가)

    -- 트랜잭션 B 시작
    START TRANSACTION;
    SELECT balance FROM accounts WHERE id = 1; -- 1000 반환
    -- 잔액을 1100으로 업데이트하려고 계산 중 (예: 100 추가)
    UPDATE accounts SET balance = 1100 WHERE id = 1;
    COMMIT;

    -- 트랜잭션 A는 계산을 마치고 변경 사항을 적용합니다:
    UPDATE accounts SET balance = 1200 WHERE id = 1;
    COMMIT;
    ```

- 보통 [pessimistic locking](/jpa/README.md#pessimistic-locking), [optimistic locking](/mysql/mysql_lock.md#mysql-optimistic-locking) 으로 해결한다. 다음은 [pessimistic locking](/jpa/README.md#pessimistic-locking) 으로 lost update 를 해결하는 예이다.

    ```sql
    -- Pessimistic Locking 사용 예시
    -- 고려해볼 상황은 두 트랜잭션이 동시에 같은 은행 계좌의 잔액을 업데이트하려고 하는 경우입니다.
    -- 트랜잭션 A와 트랜잭션 B가 accounts 테이블의 같은 행에 대해 작업을 수행합니다. 
    -- accounts 테이블에는 id, balance 필드가 있습니다.

    -- 트랜잭션 A:
    START TRANSACTION;
    -- Pessimistic Locking을 사용하여 데이터를 읽습니다.
    SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;
    -- 잔액 업데이트 작업 수행 (예: 200 추가)
    UPDATE accounts SET balance = balance + 200 WHERE id = 1;
    COMMIT;

    -- 트랜잭션 B는 트랜잭션 A가 잠금을 해제할 때까지 (즉, 트랜잭션 A가 커밋하거나 롤백할 때까지) 
    -- 해당 데이터에 대한 접근을 기다려야 합니다. 이 기다림은 트랜잭션 B가 같은 
    -- SELECT ... FOR UPDATE 쿼리를 실행할 때 발생합니다.

    -- 트랜잭션 B:
    START TRANSACTION;
    -- 트랜잭션 A가 커밋되기를 기다립니다.
    SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;
    -- 이제 트랜잭션 B는 잔액 업데이트 작업을 수행할 수 있습니다 (예: 100 추가).
    UPDATE accounts SET balance = balance + 100 WHERE id = 1;
    COMMIT;
    ```

# Serializable Anomaly

WIP...
 