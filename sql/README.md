# intro

- 주로 사용하는 phrase들을 정리해보자.

# usage

## JOIN Basic

- mysql workbench를 이용하여 다음과 같이 Customers, Order 테이블을 생성한 후 몇가지 SQL을 통해 INNER JOIN과 OUTER JOIN의 개념을 기억하자.
Customers, Orders 는 다음과 같다.

```sql
CREATE TABLE Customers (
Id INTEGER PRIMARY KEY,
Name VARCHAR(40)
);

CREATE TABLE Orders (
Id INTEGER PRIMARY KEY,
CustomerId INTEGER
);

INSERT INTO Customers VALUES(0, "foo");
INSERT INTO Customers VALUES(1, "bar");
INSERT INTO Customers VALUES(2, "buz");
INSERT INTO Orders VALUES(0, 0);
INSERT INTO Orders VALUES(1, 1);
INSERT INTO Orders VALUES(2, 1);
```

- 다음은 INNER JOIN의 결과와 같다.

```sql
SELECT * FROM Customers, Orders;
SELECT * FROM Customers JOIN Orders;
SELECT * FROM Customers INNER JOIN Orders;
```

- 다음은 LEFT OUTER JOIN이다. 왼쪽을 기준으로 오른쪽 데이터는 NULL이 가능하다.

```sql
SELECT * FROM Customers 
LEFT OUTER JOIN Orders
ON Customers.Id = Orders.CustomerId;
```

- 다음은 RIGHT OUTER JOIN이다. 오른쪽을 기준으로 왼쪽 데이터는 NULL이 가능하다.

```sql
SELECT * FROM Customers
RIGHT OUTER JOIN Orders
ON Customers.Id = Orders.CustomerId;
```

## Join ON vs WHERE

- ON은 JOIN이 실행되기 전에 적용되고 WHERE는 JOIN이 실행되고 난 다음에 적용된다.

```sql
SELECT * FROM Customers a LEFT JOIN Orders b ON (a.Id = b.Id) WHERE b.CustomerId = 1
SELECT * FROM Customers a LEFT JOIN Orders b ON (a.Id = b.Id AND b.CustomerId = 1
```

# reference

- [sqlzoo](http://sqlzoo.net/)
- [sql snippet](https://en.wikibooks.org/wiki/Structured_Query_Language/Snippets)
- [sakila](https://dev.mysql.com/doc/sakila/en/sakila-preface.html)
  - 비디오 대여점을 모델링한 example db이다. sql을 참고해서 공부하자.