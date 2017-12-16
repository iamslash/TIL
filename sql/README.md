# Intro

- 주로 사용하는 phrase들을 정리해보자.

# Material

* [use the idex luke](http://use-the-index-luke.com/)
* [sql @ w3schools](https://www.w3schools.com/sql/default.asp)

# Usage

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

## SQL Data Types (MySQL)

보통 text, number, date 로 구분한다. 

### Text

| Data type                   | storage                                                                                                                     | Description                                                                                                                                                                                                                                        |
|:---------------------------:|:---------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| CHAR(M)                     | M × w bytes, 0 <= M <= 255, where w is the number of bytes required for the maximum-length character in the character set. | Holds a fixed length string (can contain letters, numbers, and special characters). The fixed size is specified in parenthesis. Can store up to 255 characters                                                                                     |
| VARCHAR(M)                  | L + 1 bytes if column values require 0 − 255 bytes, L + 2 bytes if values may require more than 255 bytes1                  | Holds a variable length string (can contain letters, numbers, and special characters). The maximum size is specified in parenthesis. Can store up to 255 characters. Note: If you put a greater value than 255 it will be converted to a TEXT type |
| TINYTEXT                    | L + 1 bytes, where L < 2^8                                                                                                  |                                                                                                                                                                                                                                                    |
| TEXT                        | L + 2 bytes, where L < 2^16                                                                                                 |                                                                                                                                                                                                                                                    |
| BLOB                        | L + 2 bytes, where L < 2^16                                                                                                 |                                                                                                                                                                                                                                                    |
| MEDIUMTEXT                  | L + 3 bytes, where L < 2^24                                                                                                 |                                                                                                                                                                                                                                                    |
| MEDIUMBLOB                  | L + 3 bytes, where L < 2^24                                                                                                 |                                                                                                                                                                                                                                                    |
| LONGTEXT                    | L + 4 bytes, where L < 2^32                                                                                                 |                                                                                                                                                                                                                                                    |
| LONGBLOB                    | L + 4 bytes, where L < 2^32                                                                                                 |                                                                                                                                                                                                                                                    |
| ENUM('value1','value2',...) | 1 or 2 bytes, depending on the number of enumeration values (65,535 values maximum)                                         |                                                                                                                                                                                                                                                    |
| SET('value1','value2',...)  | 1, 2, 3, 4, or 8 bytes, depending on the number of set members (64 members maximum)                                         |                                                                                                                                                                                                                                                    |

### Number

| Data type       | storage | Description                                                                                                                                                                                                                           |
|:---------------:|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| TINYINT(size)   | 1       | -128 to 127 normal. 0 to 255 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                                                   |
| SMALLINT(size)  | 2       | -32768 to 32767 normal. 0 to 65535 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                                             |
| MEDIUMINT(size) | 3       | -8388608 to 8388607 normal. 0 to 16777215 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                                      |
| INT(size)       | 4       | -2147483648 to 2147483647 normal. 0 to 4294967295 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                              |
| BIGIINT(size)   | 8       | -9223372036854775808 to 9223372036854775807 normal. 0 to 18446744073709551615 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                  |
| FLOAT(size,d)   | 4,8     | A small number with a floating decimal point. The maximum number of digits may be specified in the size parameter. The maximum number of digits to the right of the decimal point is specified in the d parameter                     |
| DOUBLE(size,d)  | 8       | A large number with a floating decimal point. The maximum number of digits may be specified in the size parameter. The maximum number of digits to the right of the decimal point is specified in the d parameter                     |
| DECIMAL(size,d) | varies  | A DOUBLE stored as a string , allowing for a fixed decimal point. The maximum number of digits may be specified in the size parameter. The maximum number of digits to the right of the decimal point is specified in the d parameter |

* The integer types have an extra option called UNSIGNED. Normally,
  the integer goes from an negative to positive value. Adding the
  UNSIGNED attribute will move that range up so it starts at zero
  instead of a negative number.

### Date

| Data type | storage | Description                                                                                                                                                                                                                         |
|:---------:|:-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| DATE      | 3       | A date. Format: YYYY-MM-DD, The supported range is from '1000-01-01' to '9999-12-31'                                                                                                                                                |
| DATETIME  | 8       | A date and time combination. Format: YYYY-MM-DD HH:MI:SS, The supported range is from '1000-01-01 00:00:00' to '9999-12-31 23:59:59'                                                                                                |
| TIMESTAMP | 4       | A timestamp. TIMESTAMP values are stored as the number of seconds since the Unix epoch ('1970-01-01 00:00:00' UTC). Format: YYYY-MM-DD HH:MI:SS, The supported range is from '1970-01-01 00:00:01' UTC to '2038-01-09 03:14:07' UTC |
| TIME      | 3       | A time. Format: HH:MI:SS, The supported range is from '-838:59:59' to '838:59:59'                                                                                                                                                   |
| YEAR      | 1       | A year in two-digit or four-digit format., Values allowed in four-digit format: 1901 to 2155. Values allowed in two-digit format: 70 to 69, representing years from 1970 to 2069                                                    |

* Even if DATETIME and TIMESTAMP return the same format, they work
  very differently. In an INSERT or UPDATE query, the TIMESTAMP
  automatically set itself to the current date and time. TIMESTAMP
  also accepts various formats, like YYYYMMDDHHMISS, YYMMDDHHMISS,
  YYYYMMDD, or YYMMDD.

# References

- [sqlzoo](http://sqlzoo.net/)
- [sql snippet](https://en.wikibooks.org/wiki/Structured_Query_Language/Snippets)
- [sakila](https://dev.mysql.com/doc/sakila/en/sakila-preface.html)
  - 비디오 대여점을 모델링한 example db이다. sql을 참고해서 공부하자.
