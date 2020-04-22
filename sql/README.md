- [Intro](#intro)
- [Essentials](#essentials)
- [Materials](#materials)
- [Usage](#usage)
  - [Select statement order](#select-statement-order)
  - [Select](#select)
  - [Select Distinct](#select-distinct)
  - [Select subquery](#select-subquery)
  - [Where](#where)
  - [And, Or, Not](#and-or-not)
  - [Order By](#order-by)
  - [Insert Into](#insert-into)
  - [Null Values](#null-values)
  - [Update](#update)
  - [Delete](#delete)
  - [Select Top](#select-top)
  - [Min, Max](#min-max)
  - [Count, Avg, Sum](#count-avg-sum)
  - [Like](#like)
  - [Wildcards](#wildcards)
  - [In](#in)
  - [Between](#between)
  - [Aliases](#aliases)
  - [JOIN Basic](#join-basic)
  - [Join ON vs WHERE](#join-on-vs-where)
  - [Inner Join](#inner-join)
  - [Left Join](#left-join)
  - [Right Join](#right-join)
  - [FUll Join](#full-join)
  - [Self Join](#self-join)
  - [Union](#union)
  - [Group By](#group-by)
  - [Having](#having)
  - [Exists](#exists)
  - [Any, All](#any-all)
  - [SElect Into](#select-into)
  - [Insert Into Select](#insert-into-select)
  - [Null Functions](#null-functions)
  - [Comments](#comments)
  - [Create DB](#create-db)
  - [Drop DB](#drop-db)
  - [Create Table](#create-table)
  - [Drop Table](#drop-table)
  - [Alter Table](#alter-table)
  - [Constraints](#constraints)
  - [Not Null](#not-null)
  - [Unique](#unique)
  - [Primary Key](#primary-key)
  - [Foreign Key](#foreign-key)
  - [Check](#check)
  - [Default](#default)
  - [Index](#index)
  - [Auto Increment](#auto-increment)
  - [Dates](#dates)
  - [Views](#views)
  - [CASE](#case)
  - [Functions (MySQL)](#functions-mysql)
  - [Operators](#operators)
  - [Data Types (MySQL)](#data-types-mysql)
    - [Text](#text)
    - [Number](#number)
    - [Date](#date)
- [Problems](#problems)
- [Quiz](#quiz)

-----

# Intro

- sql 은 자주 다루는 언어가 아니기 때문에 사용법을 잊어 버리기 쉽다. [sqlzoo](http://sqlzoo.net/) 와 [database problems @ leecode](https://leetcode.com/problemset/database/) 를 복습한다.

# Essentials

- [sqlzoo](http://sqlzoo.net/)
  - tutorial 의 문제들은 필수다.
- [database problems @ leecode](https://leetcode.com/problemset/database/) 
  - 문제들을 모두 풀어보자.
* [cracking the coding interview](http://www.crackingthecodinginterview.com/)
  * databases quiz 가 볼만함

# Materials

- [db-fiddle](https://www.db-fiddle.com/)
  - sql 을 웹에서 테스트할 수 있다. 
- [sql snippet](https://en.wikibooks.org/wiki/Structured_Query_Language/Snippets)
- [sakila](https://dev.mysql.com/doc/sakila/en/sakila-preface.html)
  - 비디오 대여점을 모델링한 example db이다. sql을 참고해서 공부하자.
* [use the idex luke](http://use-the-index-luke.com/)
* [sql @ w3schools](https://www.w3schools.com/sql/default.asp)

# Usage

## Select statement order

* [SELECT 실행 순서](https://j2yes.tistory.com/entry/select%EB%AC%B8-%EC%88%9C%EC%84%9C%EC%99%80-having%EC%A0%88)

----

* order of select definition
  * SELECT
  * FROM
  * WHERE
  * GROUP BY
  * HAVING
  * ORDER BY

* order of select execution
  * FROM
  * WHERE
  * GROUP BY
  * HAVING
  * SELECT
  * ORDER BY

## Select

```sql
SELECT CustomerName, City FROM Customers;
SELECT * FROM Customers;
```

## Select Distinct

```sql
SELECT DISTINCT Country FROM Customers;
SELECT COUNT(DISTINCT Country) FROM Customers;
SELECT COUNT(*) AS DistinctCountries
  FROM (SELECT DISTINCT Country FROM Customers);
SELECT COUNT(DISTINCT Country city) FROM Customers;
```

## Select subquery

* [SQL / MySQL 서브쿼리](https://snowple.tistory.com/360)

----

subquery 는 SELECT, FROM, WHERE, HAVING, ORDER BY, VALUES of INSERT, SET of UPDATE 에 사용가능하다.

* SELECT

```sql
SELECT 
  player_name, height, (
    SELECT
      AVG(height)        
    FROM
      Player P1
    WHERE
      P1.team_id = P2.team_id
  ) AS avg_height
FROM
  Player P2
```

* FROM

```sql
SELECT 
  ROUND(AVG(ratio*100), 2) average_daily_percent
FROM
  (SELECT
    COUNT(DISTINCT(R.post_id)) / COUNT(DISTINCT(A.post_id)) AS ratio
  FROM
    Actions A
  LEFT JOIN 
    Removals R
  ON
    A.post_id = R.post_id  
  WHERE A.extra = 'spam'
  GROUP BY A.action_date) T
;
```

* WHERE

```sql
SELECT 
  student_id, 
  MIN(course_id) AS course_id, 
  grade
FROM 
  Enrollments
WHERE (student_id, grade) in
  (
    SELECT 
      student_id, 
      MAX(grade)
    FROM 
      Enrollements
    GROUP BY 
      student_id
  )
GROUP BY 
  student_id
ORDER BY 
  student_id
;
```

* HAVING

```sql
SELECT 
  project_id
FROM
  Project
GROUP BY
  project_id
HAVING
  COUNT(project_id) = (
    SELECT 
      COUNT(project_id)
    FROM
      Project
    GROUP BY
      project_id
    ORDER BY COUNT(project_id) DESC
    LIMIT 1
  )
```

## Where

```sql
SELECT * FROM Customers
  WHERE Country='Mexico';
SELECT * FROM Customers
  WHERE CustomerID=1;
```

| Operator | Description                                      |
|:--------:|:------------------------------------------------:|
| =        | Equal                                            |
| <>       | Not Equal                                        |
| >        | Greater than                                     |
| <        | Less than                                        |
| >=       | Greater than or equal                            |
| <=       | Less than or equal                               |
| BETWEEN  | Between an inclusive range                       |
| LIKE     | Search for a pattern                             |
| IN       | To specify multiple possible values for a column |

## And, Or, Not

```sql
SELECT * FROM Customers
  WHERE Country='Germany' AND City='Berlin';
SELECT * FROM Customers
  WHERE City='Berlin' OR City='München';
SELECT * FROM Customers
  WHERE NOT Country='Germany';
SELECT * FROM Customers
  WHERE Country='Germany' AND (City='Berlin' OR City='München');
SELECT * FROM Customers
  WHERE NOT Country='Germany' AND NOT Country='USA';
```

## Order By

```sql
SELECT * FROM Customers
  ORDER BY Country;
SELECT * FROM Customers
  ORDER BY Country DESC;
SELECT * FROM Customers
  ORDER BY Country, CustomerName;
SELECT * FROM Customers
  ORDER BY Country ASC, CustomerName DESC;
```

## Insert Into

```sql
INSERT INTO Customers (CustomerName, ContactName, Address, City, PostalCode, Country)
  VALUES ('Cardinal', 'Tom B. Erichsen', 'Skagen 21', 'Stavanger', '4006', 'Norway');
INSERT INTO Customers (CustomerName, City, Country)
  VALUES ('Cardinal', 'Stavanger', 'Norway');
```

## Null Values

```sql
SELECT LastName, FirstName, Address FROM Persons
  WHERE Address IS NULL;
SELECT LastName, FirstName, Address FROM Persons
  WHERE Address IS NOT NULL;
```

## Update

```sql
UPDATE Customers
  SET ContactName = 'Alfred Schmidt', City= 'Frankfurt'
  WHERE CustomerID = 1;
UPDATE Customers
  SET ContactName='Juan'
  WHERE Country='Mexico';
UPDATE Customers
  SET ContactName='Juan';
```

## Delete

```sql
DELETE FROM Customers
  WHERE CustomerName='Alfreds Futterkiste';
DELETE FROM Customers;
DELETE * FROM Customers;
```

## Select Top

```sql
SELECT TOP 3 * FROM Customers;
SELECT * FROM Customers
  LIMIT 3;
SELECT * FROM Customers
  WHERE ROWNUM <= 3;
SELECT TOP 50 PERCENT * FROM Customers;
SELECT TOP 3 * FROM Customers
  WHERE Country='Germany';
SELECT * FROM Customers
  WHERE Country='Germany'
  LIMIT 3;
SELECT * FROM Customers
  WHERE Country='Germany' AND ROWNUM <= 3;
```

## Min, Max

```sql
SELECT MIN(Price) AS SmallestPrice
  FROM Products;
SELECT MAX(Price) AS LargestPrice
  FROM Products;
```

## Count, Avg, Sum

```sql
SELECT COUNT(ProductID)
  FROM Products;
SELECT AVG(Price)
  FROM Products;
SELECT SUM(Quantity)
  FROM OrderDetails;
```

## Like

* Mysql
  * % - The percent sign represents zero, one, or multiple characters
  * _ - The underscore represents a single character


```sql
SELECT * FROM Customers
  WHERE CustomerName LIKE 'a%';
SELECT * FROM Customers
  WHERE CustomerName LIKE '%a';
SELECT * FROM Customers
  WHERE CustomerName LIKE '%or%';
SELECT * FROM Customers
  WHERE CustomerName LIKE '_r%';
SELECT * FROM Customers
  WHERE CustomerName LIKE 'a_%_%';
SELECT * FROM Customers
  WHERE ContactName LIKE 'a%o';
SELECT * FROM Customers
  WHERE CustomerName NOT LIKE 'a%';
```

## Wildcards

* Ms Access, Sql Server
  * [charlist] - Defines sets and ranges of characters to match
  * [^charlist] or [!charlist] - Defines sets and ranges of characters NOT to match

```sql
SELECT * FROM Customers
  WHERE City LIKE '[bsp]%';
SELECT * FROM Customers
  WHERE City LIKE '[a-c]%';
SELECT * FROM Customers
  WHERE City LIKE '[!bsp]%';
SELECT * FROM Customers
  WHERE City NOT LIKE '[bsp]%';
```

## In

```sql
SELECT * FROM Customers
  WHERE Country IN ('Germany', 'France', 'UK');
SELECT * FROM Customers
  WHERE Country NOT IN ('Germany', 'France', 'UK');
SELECT * FROM Customers
  WHERE Country IN (SELECT Country FROM Suppliers);
```

## Between

```sql
SELECT * FROM Products
  WHERE Price BETWEEN 10 AND 20;
SELECT * FROM Products
  WHERE Price NOT BETWEEN 10 AND 20;
SELECT * FROM Products
  WHERE (Price BETWEEN 10 AND 20)
  AND NOT CategoryID IN (1,2,3);
SELECT * FROM Products
  WHERE ProductName BETWEEN 'Carnarvon Tigers' AND 'Mozzarella di Giovanni'  
  ORDER BY ProductName;
SELECT * FROM Products
  WHERE ProductName NOT BETWEEN 'Carnarvon Tigers' AND 'Mozzarella di Giovanni'
  ORDER BY ProductName;  
SELECT * FROM Orders
  WHERE OrderDate BETWEEN #07/04/1996# AND #07/09/1996#;
```

## Aliases

```sql
SELECT CustomerID As ID, CustomerName AS Customer
  FROM Customers;
SELECT CustomerName AS Customer, ContactName AS [Contact Person]
  FROM Customers;
SELECT CustomerName, Address + ', ' + PostalCode + ' ' + City + ', ' + Country AS Address
  FROM Customers;
SELECT CustomerName, CONCAT(Address,', ',PostalCode,', ',City,', ',Country) AS Address
  FROM Customers;
SELECT o.OrderID, o.OrderDate, c.CustomerName
  FROM Customers AS c, Orders AS o
  WHERE c.CustomerName="Around the Horn" AND c.CustomerID=o.CustomerID;
SELECT Orders.OrderID, Orders.OrderDate, Customers.CustomerName
  FROM Customers, Orders
  WHERE Customers.CustomerName="Around the Horn" AND Customers.CustomerID=Orders.CustomerID;
```

## JOIN Basic

```sql
SELECT Orders.OrderID, Customers.CustomerName, Orders.OrderDate
  FROM Orders
  INNER JOIN Customers ON Orders.CustomerID=Customers.CustomerID;
```

![](img/Visual_SQL_JOINS_V2.png)

- [The JOIN operation @ sqlzoo](https://sqlzoo.net/wiki/The_JOIN_operation) 에서 연습하자.

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

## Inner Join

```sql
SELECT Orders.OrderID, Customers.CustomerName
  FROM Orders
  INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID;
SELECT Orders.OrderID, Customers.CustomerName, Shippers.ShipperName
  FROM ((Orders
           INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID)
         INNER JOIN Shippers ON Orders.ShipperID = Shippers.ShipperID);
```

## Left Join

```sql
SELECT Customers.CustomerName, Orders.OrderID
  FROM Customers
  LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID
  ORDER BY Customers.CustomerName;
```

## Right Join

```sql
SELECT Orders.OrderID, Employees.LastName, Employees.FirstName
  FROM Orders
  RIGHT JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID
  ORDER BY Orders.OrderID;
```

## FUll Join

```sql
SELECT Customers.CustomerName, Orders.OrderID
  FROM Customers
  FULL OUTER JOIN Orders ON Customers.CustomerID=Orders.CustomerID
  ORDER BY Customers.CustomerName;
```

## Self Join

```sql
SELECT A.CustomerName AS CustomerName1, B.CustomerName AS CustomerName2, A.City
  FROM Customers A, Customers B
  WHERE A.CustomerID <> B.CustomerID
  AND A.City = B.City 
  ORDER BY A.City;
```

## Union

* [UNION과 UNION ALL 의 차이 및 주의 사항](http://intomysql.blogspot.com/2011/01/union-union-all.html)


`UNION` 은 `UNION DISTINCT` 와 같다.

```sql
mysql> SELECT id, name FROM A;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foo    |
|    2 | bar    |
+------+--------+

mysql> SELECT id, name FROM B;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foo    |
|    2 | bar    |
+------+--------+

mysql> SELECT id, name FROM (
  SELECT id, name FROM A
  UNION ALL
  SELECT id, name FROM B
) x;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foo    |
|    2 | bar    |
|    1 | foo    |
|    2 | bar    |
+------+--------+


mysql> SELECT id, name FROM (
  SELECT id, name FROM A
  UNION
  SELECT id, name FROM B
) x;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foo    |
|    2 | bar    |
+------+--------+
```

```sql
SELECT City FROM Customers
  UNION
  SELECT City FROM Suppliers
  ORDER BY City;
SELECT City FROM Customers
  UNION ALL
  SELECT City FROM Suppliers
  ORDER BY City;
SELECT City, Country FROM Customers
  WHERE Country='Germany'
  UNION
  SELECT City, Country FROM Suppliers
  WHERE Country='Germany'
  ORDER BY City;
SELECT City, Country FROM Customers
  WHERE Country='Germany'
  UNION ALL
  SELECT City, Country FROM Suppliers
  WHERE Country='Germany'
  ORDER BY City;  
SELECT 'Customer' As Type, ContactName, City, Country
  FROM Customers
  UNION
  SELECT 'Supplier', ContactName, City, Country
  FROM Suppliers;  
```

## Group By

```sql
SELECT COUNT(CustomerID), Country
  FROM Customers
  GROUP BY Country;
SELECT COUNT(CustomerID), Country
  FROM Customers
  GROUP BY Country
  ORDER BY COUNT(CustomerID) DESC;
SELECT Shippers.ShipperName, COUNT(Orders.OrderID) AS NumberOfOrders FROM Orders
  LEFT JOIN Shippers ON Orders.ShipperID = Shippers.ShipperID
  GROUP BY ShipperName;
```

## Having

* The HAVING clause was added to SQL because the WHERE keyword could
  not be used with aggregate functions.

```sql
SELECT COUNT(CustomerID), Country
  FROM Customers
  GROUP BY Country
  HAVING COUNT(CustomerID) > 5;
SELECT COUNT(CustomerID), Country
  FROM Customers
  GROUP BY Country
  HAVING COUNT(CustomerID) > 5;  
SELECT COUNT(CustomerID), Country
  FROM Customers
  GROUP BY Country
  HAVING COUNT(CustomerID) > 5
  ORDER BY COUNT(CustomerID) DESC;
SELECT Employees.LastName, COUNT(Orders.OrderID) AS NumberOfOrders
  FROM (Orders
  INNER JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID)
  GROUP BY LastName
  HAVING COUNT(Orders.OrderID) > 10;
SELECT Employees.LastName, COUNT(Orders.OrderID) AS NumberOfOrders
  FROM Orders
  INNER JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID
  WHERE LastName = 'Davolio' OR LastName = 'Fuller'
  GROUP BY LastName
  HAVING COUNT(Orders.OrderID) > 25;
```

## Exists

```sql
SELECT SupplierName
  FROM Suppliers
  WHERE EXISTS (SELECT ProductName FROM Products WHERE SupplierId = Suppliers.supplierId AND Price < 20);
SELECT SupplierName
  FROM Suppliers
  WHERE EXISTS (SELECT ProductName FROM Products WHERE SupplierId = Suppliers.supplierId AND Price = 22);
```

## Any, All

```sql
SELECT ProductName
  FROM Products
  WHERE ProductID = ANY (SELECT ProductID FROM OrderDetails WHERE Quantity = 10);
SELECT ProductName
  FROM Products
  WHERE ProductID = ANY (SELECT ProductID FROM OrderDetails WHERE Quantity > 99);
SELECT ProductName
  FROM Products
  WHERE ProductID = ALL (SELECT ProductID FROM OrderDetails WHERE Quantity = 10);
```

## SElect Into

```sql
SELECT * INTO CustomersBackup2017
  FROM Customers;
SELECT * INTO CustomersBackup2017 IN 'Backup.mdb'
  FROM Customers;
SELECT CustomerName, ContactName INTO CustomersBackup2017
  FROM Customers;
SELECT * INTO CustomersGermany
  FROM Customers
  WHERE Country = 'Germany';
SELECT Customers.CustomerName, Orders.OrderID
  INTO CustomersOrderBackup2017
  FROM Customers
  LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID;
SELECT * INTO newtable
  FROM oldtable
  WHERE 1 = 0;
```

## Insert Into Select

```sql
INSERT INTO Customers (CustomerName, City, Country)
  SELECT SupplierName, City, Country FROM Suppliers;
INSERT INTO Customers (CustomerName, ContactName, Address, City, PostalCode, Country)
  SELECT SupplierName, ContactName, Address, City, PostalCode, Country FROM Suppliers;
INSERT INTO Customers (CustomerName, City, Country)
  SELECT SupplierName, City, Country FROM Suppliers
  WHERE Country='Germany';
```

## Null Functions

```sql
SELECT ProductName, UnitPrice * (UnitsInStock + IFNULL(UnitsOnOrder, 0))
  FROM Products
SELECT ProductName, UnitPrice * (UnitsInStock + COALESCE(UnitsOnOrder, 0))
  FROM Products
```

## Comments

```sql
--Select all:
SELECT * FROM Customers;

SELECT * FROM Customers -- WHERE City='Berlin';

--SELECT * FROM Customers;
SELECT * FROM Products;

/*Select all the columns
of all the records
in the Customers table:*/
SELECT * FROM Customers;

/*SELECT * FROM Customers;
SELECT * FROM Products;
SELECT * FROM Orders;
SELECT * FROM Categories;*/
SELECT * FROM Suppliers;

SELECT CustomerName, /*City,*/ Country FROM Customers;

SELECT * FROM Customers WHERE (CustomerName LIKE 'L%'
OR CustomerName LIKE 'R%' /*OR CustomerName LIKE 'S%'
OR CustomerName LIKE 'T%'*/ OR CustomerName LIKE 'W%')
AND Country='USA'
ORDER BY CustomerName;
```

## Create DB

```sql
CREATE DATABASE testDB;
```

## Drop DB

```sql
DROP DATABASE testDB;
```

## Create Table

```sql
CREATE TABLE Persons (
    PersonID int,
    LastName varchar(255),
    FirstName varchar(255),
    Address varchar(255),
    City varchar(255) 
);
```

## Drop Table

```sql
DROP TABLE Shippers;

TRUNCATE TABLE table_name;
```

## Alter Table

```sql
ALTER TABLE Persons
  ADD DateOfBirth date;

ALTER TABLE Persons
  ALTER COLUMN DateOfBirth year;

ALTER TABLE Persons
  DROP COLUMN DateOfBirth;
```

## Constraints

* NOT NULL - Ensures that a column cannot have a NULL value
* UNIQUE - Ensures that all values in a column are different
* PRIMARY KEY - A combination of a NOT NULL and UNIQUE. Uniquely identifies each row in a table
* FOREIGN KEY - Uniquely identifies a row/record in another table
* CHECK - Ensures that all values in a column satisfies a specific condition
* DEFAULT - Sets a default value for a column when no value is specified
* INDEX - Used to create and retrieve data from the database very quickly

```sql
CREATE TABLE table_name (
    column1 datatype constraint,
    column2 datatype constraint,
    column3 datatype constraint,
    ....
);
```

## Not Null

```sql
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255) NOT NULL,
    Age int
);
```

## Unique

```sql
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    UNIQUE (ID)
);

ALTER TABLE Persons
  ADD UNIQUE (ID);
ALTER table Persons DROP INDEX ID;

ALTER TABLE Persons
  ADD CONSTRAINT UC_Person UNIQUE (ID,LastName);

ALTER TABLE Persons
  DROP INDEX UC_Person;
```

## Primary Key

```sql
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (ID)
);

CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    CONSTRAINT PK_Person PRIMARY KEY (ID,LastName)
);

ALTER TABLE Persons
  ADD PRIMARY KEY (ID);

ALTER TABLE Persons
  ADD CONSTRAINT PK_Person PRIMARY KEY (ID,LastName);

ALTER TABLE Persons
  DROP PRIMARY KEY;
```

## Foreign Key

```sql
CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    PersonID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (PersonID) REFERENCES Persons(PersonID)
);

CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    PersonID int,
    PRIMARY KEY (OrderID),
    CONSTRAINT FK_PersonOrder FOREIGN KEY (PersonID)
    REFERENCES Persons(PersonID)
);

ALTER TABLE Orders
  ADD FOREIGN KEY (PersonID) REFERENCES Persons(PersonID);

ALTER TABLE Orders
  ADD CONSTRAINT FK_PersonOrder
  FOREIGN KEY (PersonID) REFERENCES Persons(PersonID);

ALTER TABLE Orders
  DROP FOREIGN KEY FK_PersonOrder;
```

## Check

```sql
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    CHECK (Age>=18)
);

CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    City varchar(255),
    CONSTRAINT CHK_Person CHECK (Age>=18 AND City='Sandnes')
);

ALTER TABLE Persons
  ADD CHECK (Age>=18);

ALTER TABLE Persons
  ADD CONSTRAINT CHK_PersonAge CHECK (Age>=18 AND City='Sandnes');

ALTER TABLE Persons
  DROP CHECK CHK_PersonAge;
```

## Default

```sql
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    City varchar(255) DEFAULT 'Sandnes'
);

CREATE TABLE Orders (
    ID int NOT NULL,
    OrderNumber int NOT NULL,
    OrderDate date DEFAULT GETDATE()
);

ALTER TABLE Persons
  ALTER City SET DEFAULT 'Sandnes';

ALTER TABLE Persons
  ALTER City DROP DEFAULT;
```

## Index

```sql
CREATE INDEX idx_lastname
  ON Persons (LastName);

CREATE INDEX idx_pname
  ON Persons (LastName, FirstName);

ALTER TABLE table_name
  DROP INDEX index_name;
```

## Auto Increment

```sql
CREATE TABLE Persons (
    ID int NOT NULL AUTO_INCREMENT,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (ID)
);

ALTER TABLE Persons AUTO_INCREMENT=100;

INSERT INTO Persons (FirstName,LastName)
VALUES ('Lars','Monsen');
```

## Dates

* DATE - format YYYY-MM-DD
* DATETIME - format: YYYY-MM-DD HH:MI:SS
* TIMESTAMP - format: YYYY-MM-DD HH:MI:SS
* YEAR - format YYYY or YY

```sql
SELECT * FROM Orders WHERE OrderDate='2008-11-11'
```

## Views

```sql
CREATE VIEW [Current Product List] AS
SELECT ProductID, ProductName
FROM Products
WHERE Discontinued = No;

SELECT * FROM [Current Product List];

CREATE VIEW [Products Above Average Price] AS
SELECT ProductName, UnitPrice
FROM Products
WHERE UnitPrice > (SELECT AVG(UnitPrice) FROM Products);

SELECT * FROM [Products Above Average Price];

CREATE VIEW [Category Sales For 1997] AS
SELECT DISTINCT CategoryName, Sum(ProductSales) AS CategorySales
FROM [Product Sales for 1997]
GROUP BY CategoryName;

SELECT * FROM [Category Sales For 1997];

SELECT * FROM [Category Sales For 1997]
  WHERE CategoryName = 'Beverages';

CREATE OR REPLACE VIEW [Current Product List] AS
SELECT ProductID, ProductName, Category
FROM Products
WHERE Discontinued = No;

DROP VIEW view_name;
```

## CASE

* `CASE WHEN THEN ELSE END` 를 기억하자.

```sql
SELECT 
  CASE 1 
    WHEN 1 
      THEN 'one'
    WHEN 2 
      THEN 'two' 
    ELSE 'more' 
  END;
# 'one'

SELECT 
  CASE 
    WHEN 1 > 0 
      THEN 'true' 
    ELSE 'false' 
  END;
# 'true'

SELECT 
  CASE BINARY 'B'
    WHEN 'a' 
      THEN 1 
    WHEN 'b' 
      THEN 2 
  END;
# NULL
```

## Functions (MySQL)

* String

```
ASCII	Returns the number code that represents the specific character
CHAR_LENGTH	Returns the length of the specified string (in characters)
CHARACTER_LENGTH	Returns the length of the specified string (in characters)
CONCAT	Concatenates two or more expressions together
CONCAT_WS	Concatenates two or more expressions together and adds a separator between them
FIELD	Returns the position of a value in a list of values
FIND_IN_SET	Returns the position of a string in a string list
FORMAT	Formats a number as a format of "#,###.##", rounding it to a certain number of decimal places
INSERT	Inserts a substring into a string at a specified position for a certain number of characters
INSTR	Returns the position of the first occurrence of a string in another string
LCASE	Converts a string to lower-case
LEFT	Extracts a substring from a string (starting from left)
LENGTH	Returns the length of the specified string (in bytes)
LOCATE	Returns the position of the first occurrence of a substring in a string
LOWER	Converts a string to lower-case
LPAD	Returns a string that is left-padded with a specified string to a certain length
LTRIM	Removes leading spaces from a string
MID	Extracts a substring from a string (starting at any position)
POSITION	Returns the position of the first occurrence of a substring in a string
REPEAT	Repeats a string a specified number of times
REPLACE	Replaces all occurrences of a specified string
REVERSE	Reverses a string and returns the result
RIGHT	Extracts a substring from a string (starting from right)
RPAD	Returns a string that is right-padded with a specified string to a certain length
RTRIM	Removes trailing spaces from a string
SPACE	Returns a string with a specified number of spaces
STRCMP	Tests whether two strings are the same
SUBSTR	Extracts a substring from a string (starting at any position)
SUBSTRING	Extracts a substring from a string (starting at any position)
SUBSTRING_INDEX	Returns the substring of string before number of occurrences of delimiter
TRIM	Removes leading and trailing spaces from a string
UCASE	Converts a string to upper-case
UPPER	Converts a string to upper-case
```

```sql
SELECT ASCII('2'); 
-- 2

SELECT CHAR_LENGTH('hello world'); 
-- 11

SELECT CONCAT('My', 'S', 'QL'); 
-- 'MySQL'
SELECT CONCAT('My', NULL, 'QL'); 
-- NULL
SELECT CONCAT(14.3); 
-- '14.3'

SELECT CONCAT_WS(',','First name','Second name','Last Name');
-- 'First name,Second name,Last Name'

SELECT CONCAT_WS(',','First name',NULL,'Last Name');
-- 'First name,Last Name'

SELECT FIELD('ej', 'Hej', 'ej', 'Heja', 'hej', 'foo');
-- 2
SELECT FIELD('fo', 'Hej', 'ej', 'Heja', 'hej', 'foo');
-- 0

SELECT FIND_IN_SET('b','a,b,c,d');
-- 2

SELECT FORMAT(12332.123456, 4);
-- '12,332.1235'
SELECT FORMAT(12332.1,4);
-- '12,332.1000'
SELECT FORMAT(12332.2,0);
-- '12,332'
SELECT FORMAT(12332.2,2,'de_DE');
-- '12.332,20'

SELECT INSERT('Quadratic', 3, 4, 'What');
-- 'QuWhattic'
SELECT INSERT('Quadratic', -1, 4, 'What');
-- 'Quadratic'
SELECT INSERT('Quadratic', 3, 100, 'What');
-- 'QuWhat'

SELECT INSTR('foobarbar', 'bar');
-- 4
SELECT INSTR('xbar', 'foobar');
-- 0

-- use LOWER instead of LCASE
SELECT LOWER('QUADRATICALLY');
        -> 'quadratically'
-- LOWER() (and UPPER()) are ineffective when applied to 
-- binary strings (BINARY, VARBINARY, BLOB). 
-- To perform lettercase conversion, 
-- convert the string to a nonbinary string:
SET @str = BINARY 'New York';
mysql> SELECT LOWER(@str), LOWER(CONVERT(@str USING latin1));
+-------------+-----------------------------------+
| LOWER(@str) | LOWER(CONVERT(@str USING latin1)) |
+-------------+-----------------------------------+
| New York    | new york                          |
+-------------+-----------------------------------+

SELECT LEFT('foobarbar', 5);
-- 'fooba'

SELECT LENGTH('text');
-- 4

SELECT LOCATE('bar', 'foobarbar');
-- 4
SELECT LOCATE('xbar', 'foobar');
-- 0
SELECT LOCATE('bar', 'foobarbar', 5);
-- 7

SELECT LPAD('hi',4,'??');
-- '??hi'
SELECT LPAD('hi',1,'??');
-- 'h'

SELECT LTRIM('  barbar');
-- 'barbar'

-- MID(str,pos,len) is a synonym for SUBSTRING(str,pos,len).

-- POSITION(substr IN str) is a synonym for LOCATE(substr,str).

SELECT REPEAT('MySQL', 3);
-- 'MySQLMySQLMySQL'

SELECT REPLACE('www.mysql.com', 'w', 'Ww');
-- 'WwWwWw.mysql.com'

SELECT REVERSE('abc');
-- 'cba'

SELECT RIGHT('foobarbar', 4);
-- 'rbar'

SELECT RPAD('hi',5,'?');
-- 'hi???'
SELECT RPAD('hi',1,'?');
-- 'h'

SELECT RTRIM('barbar   ');
-- 'barbar'

SELECT SPACE(6);
-- '      '

-- SUBSTR() is a synonym for SUBSTRING().

SELECT SUBSTRING('Quadratically',5);
-- 'ratically'
SELECT SUBSTRING('foobarbar' FROM 4);
-- 'barbar'
SELECT SUBSTRING('Quadratically',5,6);
-- 'ratica'
SELECT SUBSTRING('Sakila', -3);
-- 'ila'
SELECT SUBSTRING('Sakila', -5, 3);
-- 'aki'
SELECT SUBSTRING('Sakila' FROM -4 FOR 2);
-- 'ki'

SELECT SUBSTRING_INDEX('www.mysql.com', '.', 2);
-- 'www.mysql'
SELECT SUBSTRING_INDEX('www.mysql.com', '.', -2);
-- 'mysql.com'

SELECT TRIM('  bar   ');
-- 'bar'
SELECT TRIM(LEADING 'x' FROM 'xxxbarxxx');
-- 'barxxx'
SELECT TRIM(BOTH 'x' FROM 'xxxbarxxx');
-- 'bar'
SELECT TRIM(TRAILING 'xyz' FROM 'barxxyz');
-- 'barx'

-- UCASE() is a synonym for UPPER().

SELECT UPPER('Hej');
-- 'HEJ'
```

* Numeric

```
ABS	Returns the absolute value of a number
ACOS	Returns the arc cosine of a number
ASIN	Returns the arc sine of a number
ATAN	Returns the arc tangent of a number or the arc tangent of n and m
ATAN2	Returns the arc tangent of n and m
AVG	Returns the average value of an expression
CEIL	Returns the smallest integer value that is greater than or equal to a number
CEILING	Returns the smallest integer value that is greater than or equal to a number
COS	Returns the cosine of a number
COT	Returns the cotangent of a number
COUNT	Returns the number of records in a select query
DEGREES	Converts a radian value into degrees
DIV	Used for integer division
EXP	Returns e raised to the power of number
FLOOR	Returns the largest integer value that is less than or equal to a number
GREATEST	Returns the greatest value in a list of expressions
LEAST	Returns the smallest value in a list of expressions
LN	Returns the natural logarithm of a number
LOG	Returns the natural logarithm of a number or the logarithm of a number to a specified base
LOG10	Returns the base-10 logarithm of a number
LOG2	Returns the base-2 logarithm of a number
MAX	Returns the maximum value of an expression
MIN	Returns the minimum value of an expression
MOD	Returns the remainder of n divided by m
PI	Returns the value of PI displayed with 6 decimal places
POW	Returns m raised to the nth power
POWER	Returns m raised to the nth power
RADIANS	Converts a value in degrees to radians
RAND	Returns a random number or a random number within a range
ROUND	Returns a number rounded to a certain number of decimal places
SIGN	Returns a value indicating the sign of a number
SIN	Returns the sine of a number
SQRT	Returns the square root of a number
SUM	Returns the summed value of an expression
TAN	Returns the tangent of a number
TRUNCATE	Returns a number truncated to a certain number of decimal places
```

```sql
SELECT ABS(2);
-- 2
SELECT ABS(-32);
-- 32

SELECT ACOS(1);
-- 0
SELECT ACOS(1.0001);
-- NULL
SELECT ACOS(0);
-- 1.5707963267949

SELECT ASIN(0.2);
-- 0.20135792079033
SELECT ASIN('foo');

+-------------+
| ASIN('foo') |
+-------------+
|           0 |
+-------------+
1 row in set, 1 warning (0.00 sec)

SHOW WARNINGS;
+---------+------+-----------------------------------------+
| Level   | Code | Message                                 |
+---------+------+-----------------------------------------+
| Warning | 1292 | Truncated incorrect DOUBLE value: 'foo' |
+---------+------+-----------------------------------------+

SELECT ATAN(-2,2);
-- -0.78539816339745
SELECT ATAN2(PI(),0);
-- 1.5707963267949

-- CEIL() is a synonym for CEILING().

SELECT CEILING(1.23);
-- 2
SELECT CEILING(-1.23);
-- -1

SELECT COS(PI());
-- -1

SELECT COT(12);
-- -1.5726734063977
SELECT COT(0);
-- out-of-range error

SELECT DEGREES(PI());
-- 180
SELECT DEGREES(PI() / 2);
-- 90

SELECT EXP(2);
-- 7.3890560989307
SELECT EXP(-2);
-- 0.13533528323661
SELECT EXP(0);
-- 1

SELECT FLOOR(1.23), FLOOR(-1.23);
-- 1, -2

SELECT LOG(2);
-- 0.69314718055995
SELECT LOG(-2);
-- NULL

SELECT LOG2(65536);
-- 16
SELECT LOG2(-100);
-- NULL

SELECT LOG10(2);
-- 0.30102999566398
SELECT LOG10(100);
-- 2
SELECT LOG10(-100);
-- NULL

SELECT MOD(234, 10);
-- 4
SELECT 253 % 7;
-- 1
SELECT MOD(29,9);
-- 2
SELECT 29 MOD 9;
-- 2
SELECT MOD(34.5,3);
-- 1.5

SELECT PI();
-- 3.141593
SELECT PI()+0.000000000000000000;
-- 3.141592653589793116

SELECT POW(2,2);
-- 4
SELECT POW(2,-2);
-- 0.25

-- POWER() is a synonym for POW().

SELECT RADIANS(90);
-- 1.5707963267949

SELECT FLOOR(7 + (RAND() * 5));

mysql> CREATE TABLE t (i INT);
Query OK, 0 rows affected (0.42 sec)

mysql> INSERT INTO t VALUES(1),(2),(3);
Query OK, 3 rows affected (0.00 sec)
Records: 3  Duplicates: 0  Warnings: 0

mysql> SELECT i, RAND() FROM t;
+------+------------------+
| i    | RAND()           |
+------+------------------+
|    1 | 0.61914388706828 |
|    2 | 0.93845168309142 |
|    3 | 0.83482678498591 |
+------+------------------+
3 rows in set (0.00 sec)

mysql> SELECT i, RAND(3) FROM t;
+------+------------------+
| i    | RAND(3)          |
+------+------------------+
|    1 | 0.90576975597606 |
|    2 | 0.37307905813035 |
|    3 | 0.14808605345719 |
+------+------------------+
3 rows in set (0.00 sec)

mysql> SELECT i, RAND() FROM t;
+------+------------------+
| i    | RAND()           |
+------+------------------+
|    1 | 0.35877890638893 |
|    2 | 0.28941420772058 |
|    3 | 0.37073435016976 |
+------+------------------+
3 rows in set (0.00 sec)

mysql> SELECT i, RAND(3) FROM t;
+------+------------------+
| i    | RAND(3)          |
+------+------------------+
|    1 | 0.90576975597606 |
|    2 | 0.37307905813035 |
|    3 | 0.14808605345719 |
+------+------------------+
3 rows in set (0.01 sec)

mysql> SELECT ROUND(-1.23);
        -> -1
mysql> SELECT ROUND(-1.58);
        -> -2
mysql> SELECT ROUND(1.58);
        -> 2
mysql> SELECT ROUND(1.298, 1);
        -> 1.3
mysql> SELECT ROUND(1.298, 0);
        -> 1
mysql> SELECT ROUND(23.298, -1);
        -> 20
mysql> SELECT ROUND(150.000,2), ROUND(150,2);
+------------------+--------------+
| ROUND(150.000,2) | ROUND(150,2) |
+------------------+--------------+
|           150.00 |          150 |
+------------------+--------------+        
mysql> SELECT ROUND(2.5), ROUND(25E-1);
+------------+--------------+
| ROUND(2.5) | ROUND(25E-1) |
+------------+--------------+
| 3          |            2 |
+------------+--------------+        
        
mysql> SELECT SIGN(-32);
        -> -1
mysql> SELECT SIGN(0);
        -> 0
mysql> SELECT SIGN(234);
        -> 1
        
mysql> SELECT SIN(PI());
        -> 1.2246063538224e-16
mysql> SELECT ROUND(SIN(PI()));
        -> 0
        
mysql> SELECT SQRT(4);
        -> 2
mysql> SELECT SQRT(20);
        -> 4.4721359549996
mysql> SELECT SQRT(-16);
        -> NULL
        
mysql> SELECT TAN(PI());
        -> -1.2246063538224e-16
mysql> SELECT TAN(PI()+1);
        -> 1.5574077246549        

mysql> SELECT TRUNCATE(1.223,1);
        -> 1.2
mysql> SELECT TRUNCATE(1.999,1);
        -> 1.9
mysql> SELECT TRUNCATE(1.999,0);
        -> 1
mysql> SELECT TRUNCATE(-1.999,1);
        -> -1.9
mysql> SELECT TRUNCATE(122,-2);
       -> 100
mysql> SELECT TRUNCATE(10.28*100,0);
       -> 1028
```

* Date 

```
ADDDATE	Returns a date after a certain time/date interval has been added
ADDTIME	Returns a time/datetime after a certain time interval has been added
CURDATE	Returns the current date
CURRENT_DATE	Returns the current date
CURRENT_TIME	Returns the current time
CURRENT_TIMESTAMP	Returns the current date and time
CURTIME	Returns the current time
DATE	Extracts the date value from a date or datetime expression
DATEDIFF	Returns the difference in days between two date values
DATE_ADD	Returns a date after a certain time/date interval has been added
DATE_FORMAT	Formats a date as specified by a format mask
DATE_SUB	Returns a date after a certain time/date interval has been subtracted
DAY	Returns the day portion of a date value
DAYNAME	Returns the weekday name for a date
DAYOFMONTH	Returns the day portion of a date value
DAYOFWEEK	Returns the weekday index for a date value
DAYOFYEAR	Returns the day of the year for a date value
EXTRACT	Extracts parts from a date
FROM_DAYS	Returns a date value from a numeric representation of the day
HOUR	Returns the hour portion of a date value
LAST_DAY	Returns the last day of the month for a given date
LOCALTIME	Returns the current date and time
LOCALTIMESTAMP	Returns the current date and time
MAKEDATE	Returns the date for a certain year and day-of-year value
MAKETIME	Returns the time for a certain hour, minute, second combination
MICROSECOND	Returns the microsecond portion of a date value
MINUTE	Returns the minute portion of a date value
MONTH	Returns the month portion of a date value
MONTHNAME	Returns the full month name for a date
NOW	Returns the current date and time
PERIOD_ADD	Takes a period and adds a specified number of months to it
PERIOD_DIFF	Returns the difference in months between two periods
QUARTER	Returns the quarter portion of a date value
SECOND	Returns the second portion of a date value
SEC_TO_TIME	Converts numeric seconds into a time value
STR_TO_DATE	Takes a string and returns a date specified by a format mask
SUBDATE	Returns a date after which a certain time/date interval has been subtracted
SUBTIME	Returns a time/datetime value after a certain time interval has been subtracted
SYSDATE	Returns the current date and time
TIME	Extracts the time value from a time/datetime expression
TIME_FORMAT	Formats a time as specified by a format mask
TIME_TO_SEC	Converts a time value into numeric seconds
TIMEDIFF	Returns the difference between two time/datetime values
TIMESTAMP	Converts an expression to a datetime value and if specified adds an optional time interval to the value
TO_DAYS	Converts a date into numeric days
WEEK	Returns the week portion of a date value
WEEKDAY	Returns the weekday index for a date value
WEEKOFYEAR	Returns the week of the year for a date value
YEAR	Returns the year portion of a date value
YEARWEEK	Returns the year and week for a date value
```

```sql
mysql> SELECT DATE_ADD('2008-01-02', INTERVAL 31 DAY);
        -> '2008-02-02'
mysql> SELECT ADDDATE('2008-01-02', INTERVAL 31 DAY);
        -> '2008-02-02'
mysql> SELECT ADDDATE('2008-01-02', 31);
        -> '2008-02-02'
        
mysql> SELECT ADDTIME('2007-12-31 23:59:59.999999', '1 1:1:1.000002');
        -> '2008-01-02 01:01:01.000001'
mysql> SELECT ADDTIME('01:00:00.999999', '02:00:00.999998');
        -> '03:00:01.999997'
        
mysql> SELECT CURDATE();
        -> '2008-06-13'
mysql> SELECT CURDATE() + 0;
        -> 20080613        

-- CURRENT_DATE and CURRENT_DATE() are synonyms for CURDATE().
-- CURRENT_TIME and CURRENT_TIME() are synonyms for CURTIME().        
-- CURRENT_TIMESTAMP and CURRENT_TIMESTAMP() are synonyms for NOW().

mysql> SELECT CURTIME();
        -> '23:50:26'
mysql> SELECT CURTIME() + 0;
        -> 235026.000000

mysql> SELECT DATE('2003-12-31 01:02:03');
        -> '2003-12-31'

mysql> SELECT DATEDIFF('2007-12-31 23:59:59','2007-12-30');
        -> 1
mysql> SELECT DATEDIFF('2010-11-30 23:59:59','2010-12-31');
        -> -31

mysql> SELECT '2008-12-31 23:59:59' + INTERVAL 1 SECOND;
        -> '2009-01-01 00:00:00'
mysql> SELECT INTERVAL 1 DAY + '2008-12-31';
        -> '2009-01-01'
mysql> SELECT '2005-01-01' - INTERVAL 1 SECOND;
        -> '2004-12-31 23:59:59'
mysql> SELECT DATE_ADD('2000-12-31 23:59:59',
    ->                 INTERVAL 1 SECOND);
        -> '2001-01-01 00:00:00'
mysql> SELECT DATE_ADD('2010-12-31 23:59:59',
    ->                 INTERVAL 1 DAY);
        -> '2011-01-01 23:59:59'
mysql> SELECT DATE_ADD('2100-12-31 23:59:59',
    ->                 INTERVAL '1:1' MINUTE_SECOND);
        -> '2101-01-01 00:01:00'
mysql> SELECT DATE_SUB('2005-01-01 00:00:00',
    ->                 INTERVAL '1 1:1:1' DAY_SECOND);
        -> '2004-12-30 22:58:59'
mysql> SELECT DATE_ADD('1900-01-01 00:00:00',
    ->                 INTERVAL '-1 10' DAY_HOUR);
        -> '1899-12-30 14:00:00'
mysql> SELECT DATE_SUB('1998-01-02', INTERVAL 31 DAY);
        -> '1997-12-02'
mysql> SELECT DATE_ADD('1992-12-31 23:59:59.000002',
    ->            INTERVAL '1.999999' SECOND_MICROSECOND);
        -> '1993-01-01 00:00:01.000001'
        
mysql> SELECT 6/4;
        -> 1.5000
mysql> SELECT DATE_ADD('2009-01-01', INTERVAL 6/4 HOUR_MINUTE);
        -> '2009-01-04 12:20:00'
        
mysql> SELECT CAST(6/4 AS DECIMAL(3,1));
        -> 1.5
mysql> SELECT DATE_ADD('1970-01-01 12:00:00',
    ->                 INTERVAL CAST(6/4 AS DECIMAL(3,1)) HOUR_MINUTE);
        -> '1970-01-01 13:05:00'
        
mysql> SELECT DATE_ADD('2013-01-01', INTERVAL 1 DAY);
        -> '2013-01-02'
mysql> SELECT DATE_ADD('2013-01-01', INTERVAL 1 HOUR);
        -> '2013-01-01 01:00:00'
        
mysql> SELECT DATE_ADD('2009-01-30', INTERVAL 1 MONTH);
        -> '2009-02-28'

mysql> SELECT DATE_ADD('2006-07-00', INTERVAL 1 DAY);
        -> NULL
mysql> SELECT '2005-03-32' + INTERVAL 1 MONTH;
        -> NULL
        
mysql> SELECT DATE_FORMAT('2009-10-04 22:23:00', '%W %M %Y');
        -> 'Sunday October 2009'
mysql> SELECT DATE_FORMAT('2007-10-04 22:23:00', '%H:%i:%s');
        -> '22:23:00'
mysql> SELECT DATE_FORMAT('1900-10-04 22:23:00',
    ->                 '%D %y %a %d %m %b %j');
        -> '4th 00 Thu 04 10 Oct 277'
mysql> SELECT DATE_FORMAT('1997-10-04 22:23:00',
    ->                 '%H %k %I %r %T %S %w');
        -> '22 22 10 10:23:00 PM 22:23:00 00 6'
mysql> SELECT DATE_FORMAT('1999-01-01', '%X %V');
        -> '1998 52'
mysql> SELECT DATE_FORMAT('2006-06-00', '%d');
        -> '00'

-- DAY() is a synonym for DAYOFMONTH().

mysql> SELECT DAYNAME('2007-02-03');
        -> 'Saturday'

mysql> SELECT DAYOFMONTH('2007-02-03');
        -> 3
    
mysql> SELECT DAYOFWEEK('2007-02-03');
        -> 7
        
mysql> SELECT DAYOFYEAR('2007-02-03');
        -> 34
        
mysql> SELECT EXTRACT(YEAR FROM '2009-07-02');
       -> 2009
mysql> SELECT EXTRACT(YEAR_MONTH FROM '2009-07-02 01:02:03');
       -> 200907
mysql> SELECT EXTRACT(DAY_MINUTE FROM '2009-07-02 01:02:03');
       -> 20102
mysql> SELECT EXTRACT(MICROSECOND
    ->                FROM '2003-01-02 10:30:00.000123');
        -> 123

mysql> SELECT FROM_DAYS(730669);
        -> '2000-07-03'

mysql> SELECT HOUR('10:05:03');
        -> 10
mysql> SELECT HOUR('272:59:59');
        -> 272
        
mysql> SELECT LAST_DAY('2003-02-05');
        -> '2003-02-28'
mysql> SELECT LAST_DAY('2004-02-05');
        -> '2004-02-29'
mysql> SELECT LAST_DAY('2004-01-01 01:01:01');
        -> '2004-01-31'
mysql> SELECT LAST_DAY('2003-03-32');
        -> NULL
        
-- LOCALTIME and LOCALTIME() are synonyms for NOW().
-- LOCALTIMESTAMP and LOCALTIMESTAMP() are synonyms for NOW().

mysql> SELECT MAKEDATE(2011,31), MAKEDATE(2011,32);
        -> '2011-01-31', '2011-02-01'
mysql> SELECT MAKEDATE(2011,365), MAKEDATE(2014,365);
        -> '2011-12-31', '2014-12-31'
mysql> SELECT MAKEDATE(2011,0);
        -> NULL
        
mysql> SELECT MAKETIME(12,15,30);
        -> '12:15:30'

mysql> SELECT MICROSECOND('12:00:00.123456');
        -> 123456
mysql> SELECT MICROSECOND('2009-12-31 23:59:59.000010');
        -> 10

mysql> SELECT MINUTE('2008-02-03 10:05:03');
        -> 5

mysql> SELECT MONTH('2008-02-03');
        -> 2
        
mysql> SELECT MONTHNAME('2008-02-03');
        -> 'February'
        
mysql> SELECT NOW();
        -> '2007-12-15 23:50:26'
mysql> SELECT NOW() + 0;
        -> 20071215235026.000000
        
mysql> SELECT NOW(), SLEEP(2), NOW();
+---------------------+----------+---------------------+
| NOW()               | SLEEP(2) | NOW()               |
+---------------------+----------+---------------------+
| 2006-04-12 13:47:36 |        0 | 2006-04-12 13:47:36 |
+---------------------+----------+---------------------+

mysql> SELECT SYSDATE(), SLEEP(2), SYSDATE();
+---------------------+----------+---------------------+
| SYSDATE()           | SLEEP(2) | SYSDATE()           |
+---------------------+----------+---------------------+
| 2006-04-12 13:47:44 |        0 | 2006-04-12 13:47:46 |
+---------------------+----------+---------------------+

mysql> SELECT PERIOD_ADD(200801,2);
        -> 200803
        
mysql> SELECT PERIOD_DIFF(200802,200703);
        -> 11
        
mysql> SELECT QUARTER('2008-04-01');
        -> 2
        
mysql> SELECT SECOND('10:05:03');
        -> 3
       
mysql> SELECT SEC_TO_TIME(2378);
        -> '00:39:38'
mysql> SELECT SEC_TO_TIME(2378) + 0;
        -> 3938
        
mysql> SELECT STR_TO_DATE('01,5,2013','%d,%m,%Y');
        -> '2013-05-01'
mysql> SELECT STR_TO_DATE('May 1, 2013','%M %d,%Y');
        -> '2013-05-01'
        
mysql> SELECT STR_TO_DATE('a09:30:17','a%h:%i:%s');
        -> '09:30:17'
mysql> SELECT STR_TO_DATE('a09:30:17','%h:%i:%s');
        -> NULL
mysql> SELECT STR_TO_DATE('09:30:17a','%h:%i:%s');
        -> '09:30:17'

mysql> SELECT STR_TO_DATE('abc','abc');
        -> '0000-00-00'
mysql> SELECT STR_TO_DATE('9','%m');
        -> '0000-09-00'
mysql> SELECT STR_TO_DATE('9','%s');
        -> '00:00:09'
        
mysql> SELECT STR_TO_DATE('00/00/0000', '%m/%d/%Y');
        -> '0000-00-00'
mysql> SELECT STR_TO_DATE('04/31/2004', '%m/%d/%Y');
        -> '2004-04-31'
        
mysql> SET sql_mode = '';
mysql> SELECT STR_TO_DATE('15:35:00', '%H:%i:%s');
+-------------------------------------+
| STR_TO_DATE('15:35:00', '%H:%i:%s') |
+-------------------------------------+
| 15:35:00                            |
+-------------------------------------+
mysql> SET sql_mode = 'NO_ZERO_IN_DATE';
mysql> SELECT STR_TO_DATE('15:35:00', '%h:%i:%s');
+-------------------------------------+
| STR_TO_DATE('15:35:00', '%h:%i:%s') |
+-------------------------------------+
| NULL                                |
+-------------------------------------+
mysql> SHOW WARNINGS\G
*************************** 1. row ***************************
  Level: Warning
   Code: 1411
Message: Incorrect datetime value: '15:35:00' for function str_to_date

mysql> SELECT DATE_SUB('2008-01-02', INTERVAL 31 DAY);
        -> '2007-12-02'
mysql> SELECT SUBDATE('2008-01-02', INTERVAL 31 DAY);
        -> '2007-12-02'

mysql> SELECT SUBDATE('2008-01-02 12:00:00', 31);
        -> '2007-12-02 12:00:00'

mysql> SELECT SUBTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002');
        -> '2007-12-30 22:58:58.999997'
mysql> SELECT SUBTIME('01:00:00.999999', '02:00:00.999998');
        -> '-00:59:59.999999'
        
mysql> SELECT NOW(), SLEEP(2), NOW();
+---------------------+----------+---------------------+
| NOW()               | SLEEP(2) | NOW()               |
+---------------------+----------+---------------------+
| 2006-04-12 13:47:36 |        0 | 2006-04-12 13:47:36 |
+---------------------+----------+---------------------+

mysql> SELECT SYSDATE(), SLEEP(2), SYSDATE();
+---------------------+----------+---------------------+
| SYSDATE()           | SLEEP(2) | SYSDATE()           |
+---------------------+----------+---------------------+
| 2006-04-12 13:47:44 |        0 | 2006-04-12 13:47:46 |
+---------------------+----------+---------------------+

mysql> SELECT TIME('2003-12-31 01:02:03');
        -> '01:02:03'
mysql> SELECT TIME('2003-12-31 01:02:03.000123');
        -> '01:02:03.000123'

mysql> SELECT TIMEDIFF('2000:01:01 00:00:00',
    ->                 '2000:01:01 00:00:00.000001');
        -> '-00:00:00.000001'
mysql> SELECT TIMEDIFF('2008-12-31 23:59:59.000001',
    ->                 '2008-12-30 01:01:01.000002');
        -> '46:58:57.999999'
        
mysql> SELECT TIMESTAMP('2003-12-31');
        -> '2003-12-31 00:00:00'
mysql> SELECT TIMESTAMP('2003-12-31 12:00:00','12:00:00');
        -> '2004-01-01 00:00:00'
        
mysql> SELECT TO_DAYS(950501);
        -> 728779
mysql> SELECT TO_DAYS('2007-10-07');
        -> 733321
        
mysql> SELECT TO_DAYS('2008-10-07'), TO_DAYS('08-10-07');
        -> 733687, 733687
        
mysql> SELECT TO_DAYS('0000-00-00');
+-----------------------+
| to_days('0000-00-00') |
+-----------------------+
|                  NULL |
+-----------------------+
1 row in set, 1 warning (0.00 sec)

mysql> SHOW WARNINGS;
+---------+------+----------------------------------------+
| Level   | Code | Message                                |
+---------+------+----------------------------------------+
| Warning | 1292 | Incorrect datetime value: '0000-00-00' |
+---------+------+----------------------------------------+
1 row in set (0.00 sec)


mysql> SELECT TO_DAYS('0000-01-01');
+-----------------------+
| to_days('0000-01-01') |
+-----------------------+
|                     1 |
+-----------------------+
1 row in set (0.00 sec)

mysql> SELECT WEEK('2008-02-20');
        -> 7
mysql> SELECT WEEK('2008-02-20',0);
        -> 7
mysql> SELECT WEEK('2008-02-20',1);
        -> 8
mysql> SELECT WEEK('2008-12-31',1);
        -> 53
        
mysql> SELECT YEAR('2000-01-01'), WEEK('2000-01-01',0);
        -> 2000, 0

mysql> SELECT WEEK('2000-01-01',2);
        -> 52

mysql> SELECT YEAR('2000-01-01'), WEEK('2000-01-01',0);
        -> 2000, 0
        
mysql> SELECT WEEK('2000-01-01',2);
        -> 52
        
mysql> SELECT YEARWEEK('2000-01-01');
        -> 199952
mysql> SELECT MID(YEARWEEK('2000-01-01'),5,2);
        -> '52'
        
mysql> SELECT WEEKDAY('2008-02-03 22:23:00');
        -> 6
mysql> SELECT WEEKDAY('2007-11-06');
        -> 1
        
mysql> SELECT WEEKOFYEAR('2008-02-20');
        -> 8
        
mysql> SELECT YEAR('1987-01-01');
        -> 1987
        
mysql> SELECT YEARWEEK('1987-01-01');
        -> 198652
```

* Advanced

```
BIN	Converts a decimal number to a binary number
BINARY	Converts a value to a binary string
CASE	Lets you evaluate conditions and return a value when the first condition is met
CAST	Converts a value from one datatype to another datatype
COALESCE	Returns the first non-null expression in a list
CONNECTION_ID	Returns the unique connection ID for the current connection
CONV	Converts a number from one number base to another
CONVERT	Converts a value from one datatype to another, or one character set to another
CURRENT_USER	Returns the user name and host name for the MySQL account used by the server to authenticate the current client
DATABASE	Returns the name of the default database
IF	Returns one value if a condition is TRUE, or another value if a condition is FALSE
IFNULL	Lets you to return an alternate value if an expression is NULL
ISNULL	Tests whether an expression is NULL
LAST_INSERT_ID	Returns the first AUTO_INCREMENT value that was set by the most recent INSERT or UPDATE statement
NULLIF	Compares two expressions
SESSION_USER	Returns the user name and host name for the current MySQL user
SYSTEM_USER	Returns the user name and host name for the current MySQL user
USER	Returns the user name and host name for the current MySQL user
VERSION	Returns the version of the MySQL database
```

```sql
mysql> SELECT BIN(12);
        -> '1100'

mysql> SELECT 'a' = 'A';
        -> 1
mysql> SELECT BINARY 'a' = 'A';
        -> 0
mysql> SELECT 'a' = 'a ';
        -> 1
mysql> SELECT BINARY 'a' = 'a ';
        -> 0

-- The CAST() function takes an expression of any type and produces a result value of the specified type, similar to CONVERT(). 

mysql> SELECT CASE 1 WHEN 1 THEN 'one'
    ->     WHEN 2 THEN 'two' ELSE 'more' END;
        -> 'one'
mysql> SELECT CASE WHEN 1>0 THEN 'true' ELSE 'false' END;
        -> 'true'
mysql> SELECT CASE BINARY 'B'
    ->     WHEN 'a' THEN 1 WHEN 'b' THEN 2 END;
        -> NULL
        
mysql> SELECT COALESCE(NULL,1);
        -> 1
mysql> SELECT COALESCE(NULL,NULL,NULL);
        -> NULL

mysql> SELECT CONNECTION_ID();
        -> 23786
        
mysql> SELECT CONV('a',16,2);
        -> '1010'
mysql> SELECT CONV('6E',18,8);
        -> '172'
mysql> SELECT CONV(-17,10,-18);
        -> '-H'
mysql> SELECT CONV(10+'10'+'10'+X'0a',10,10);
        -> '40'

SELECT CONVERT('abc' USING utf8);

mysql> SELECT USER();
        -> 'davida@localhost'
mysql> SELECT * FROM mysql.user;
ERROR 1044: Access denied for user ''@'localhost' to
database 'mysql'
mysql> SELECT CURRENT_USER();
        -> '@localhost'
        
mysql> SELECT DATABASE();
        -> 'test'
        
mysql> SELECT LAST_INSERT_ID();
        -> 195
        
mysql> USE test;

mysql> CREATE TABLE t (
       id INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
       name VARCHAR(10) NOT NULL
       );

mysql> INSERT INTO t VALUES (NULL, 'Bob');

mysql> SELECT * FROM t;
+----+------+
| id | name |
+----+------+
|  1 | Bob  |
+----+------+

mysql> SELECT LAST_INSERT_ID();
+------------------+
| LAST_INSERT_ID() |
+------------------+
|                1 |
+------------------+

mysql> INSERT INTO t VALUES
       (NULL, 'Mary'), (NULL, 'Jane'), (NULL, 'Lisa');

mysql> SELECT * FROM t;
+----+------+
| id | name |
+----+------+
|  1 | Bob  |
|  2 | Mary |
|  3 | Jane |
|  4 | Lisa |
+----+------+

mysql> SELECT LAST_INSERT_ID();
+------------------+
| LAST_INSERT_ID() |
+------------------+
|                2 |
+------------------+

mysql> SELECT IF(1>2,2,3);
        -> 3
mysql> SELECT IF(1<2,'yes','no');
        -> 'yes'
mysql> SELECT IF(STRCMP('test','test1'),'no','yes');
        -> 'no'
        
mysql> SELECT IFNULL(1,0);
        -> 1
mysql> SELECT IFNULL(NULL,10);
        -> 10
mysql> SELECT IFNULL(1/0,10);
        -> 10
mysql> SELECT IFNULL(1/0,'yes');
        -> 'yes'
        
mysql> SELECT 1 IS NULL, 0 IS NULL, NULL IS NULL;
        -> 0, 0, 1
        
-- SESSION_USER() is a synonym for USER().
-- SYSTEM_USER() is a synonym for USER().

mysql> SELECT USER();
        -> 'davida@localhost'

mysql> SELECT VERSION();
        -> '5.7.22-standard'
```

## Operators

* Arithmetic

| Operator | Description |
|:--------:|:-----------:|
| +        | Add         |
| -        | Sub         |
| *        | Mul         |
| /        | Div         |
| %        | Modulo      |

* Bitwise

| Operator | Description |
|:--------:|:-----------:|
| &        | AND         |
|          | OR          |
| ^        | XOR         |

* Comparison

| Operator | Description           |
|:--------:|:---------------------:|
| =        | Equal                 |
| <>       | Not Equal             |
| >        | Greater than          |
| <        | Less than             |
| >=       | Greater than or equal |
| <=       | Less than or equal    |

* Compound

| Operator | Description    |
|:--------:|:--------------:|
| +=       | Add equals     |
| -=       | Sub equals     |
| *=       | multiply equal |
| /=       | div equal      |
| %=       | modulo equal   |
| &=       | AND equal      |
| ^-=      | XOR equal      |
|          | OR equal       |

* Logical

| Operator | Description                                                  |
|:--------:|:------------------------------------------------------------:|
| ALL      | TRUE if all of the subquery values meet the condition        |
| AND      | TRUE if all the conditions separated by AND is TRUE          |
| ANY      | TRUE if any of the subquery values meet the condition        |
| BETWEEN  | TRUE if the operand is within the range of comparisons       |
| EXISTS   | TRUE if the subquery returns one or more records             |
| IN       | TRUE if the operand is equal to one of a list of expressions |
| LIKE     | TRUE if the operand matches a pattern                        |
| NOT      | Displays a record if the condition(s) is NOT TRUE            |
| OR       | TRUE if any of the conditions separated by OR is TRUE        |
| SOME     | TRUE if any of the subquery values meet the condition        |

## Data Types (MySQL)

보통 text, number, date 로 구분한다. 

### Text

| Data type                   | storage                                                                                                                     | Description                                                                                                                                                                                                                                                                                        |
|:---------------------------:|:---------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| CHAR(M)                     | M × w bytes, 0 <= M <= 255, where w is the number of bytes required for the maximum-length character in the character set. | Holds a fixed length string (can contain letters, numbers, and special characters). The fixed size is specified in parenthesis. Can store up to 255 characters                                                                                                                                     |
| VARCHAR(M)                  | L + 1 bytes if column values require 0 − 255 bytes, L + 2 bytes if values may require more than 255 bytes1                  | Holds a variable length string (can contain letters, numbers, and special characters). The maximum size is specified in parenthesis. Can store up to 255 characters. Note: If you put a greater value than 255 it will be converted to a TEXT type                                                 |
| TINYTEXT                    | L + 1 bytes, where L < 2^8                                                                                                  | Holds a string with a maximum length of 255 characters                                                                                                                                                                                                                                             |
| TEXT                        | L + 2 bytes, where L < 2^16                                                                                                 | Holds a string with a maximum length of 65,535 characters                                                                                                                                                                                                                                          |
| BLOB                        | L + 2 bytes, where L < 2^16                                                                                                 | For BLOBs (Binary Large OBjects). Holds up to 65,535 bytes of data                                                                                                                                                                                                                                 |
| MEDIUMTEXT                  | L + 3 bytes, where L < 2^24                                                                                                 | Holds a string with a maximum length of 16,777,215 characters                                                                                                                                                                                                                                      |
| MEDIUMBLOB                  | L + 3 bytes, where L < 2^24                                                                                                 | For BLOBs (Binary Large OBjects). Holds up to 16,777,215 bytes of data                                                                                                                                                                                                                             |
| LONGTEXT                    | L + 4 bytes, where L < 2^32                                                                                                 | Holds a string with a maximum length of 4,294,967,295 characters                                                                                                                                                                                                                                   |
| LONGBLOB                    | L + 4 bytes, where L < 2^32                                                                                                 | For BLOBs (Binary Large OBjects). Holds up to 4,294,967,295 bytes of data                                                                                                                                                                                                                          |
| ENUM('value1','value2',...) | 1 or 2 bytes, depending on the number of enumeration values (65,535 values maximum)                                         | Let you enter a list of possible values. You can list up to 65535 values in an ENUM list. If a value is inserted that is not in the list, a blank value will be inserted. Note: The values are sorted in the order you enter them. You enter the possible values in this format: ENUM('X','Y','Z') |
| SET('value1','value2',...)  | 1, 2, 3, 4, or 8 bytes, depending on the number of set members (64 members maximum)                                         | Similar to ENUM except that SET may contain up to 64 list items and can store more than one choice                                                                                                                                                                                                 |

### Number

| Data type    | storage                                           | Description                                                                                                                                                                                                                           |
|:------------:|:-------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| TINYINT(M)   | 1                                                 | -128 to 127 normal. 0 to 255 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                                                   |
| SMALLINT(M)  | 2                                                 | -32768 to 32767 normal. 0 to 65535 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                                             |
| MEDIUMINT(M) | 3                                                 | -8388608 to 8388607 normal. 0 to 16777215 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                                      |
| INT(M)       | 4                                                 | -2147483648 to 2147483647 normal. 0 to 4294967295 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                                              |
| BIGIINT(M)   | 8                                                 | -9223372036854775808 to 9223372036854775807 normal. 0 to 18446744073709551615 UNSIGNED. The maximum number of digits may be specified in parenthesis                                                                                  |
| FLOAT(M,d)   | 4 bytes if 0 <= p <= 24, 8 bytes if 25 <= p <= 53 | A small number with a floating decimal point. The maximum number of digits may be specified in the size parameter. The maximum number of digits to the right of the decimal point is specified in the d parameter                     |
| DOUBLE(M,d)  | 8                                                 | A large number with a floating decimal point. The maximum number of digits may be specified in the size parameter. The maximum number of digits to the right of the decimal point is specified in the d parameter                     |
| DECIMAL(M,d) | total M digits, fraction d digits                 | A DOUBLE stored as a string , allowing for a fixed decimal point. The maximum number of digits may be specified in the size parameter. The maximum number of digits to the right of the decimal point is specified in the d parameter |

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

# Problems

* `WHERE IN`
  * [Highest Grade For Each Student @ learntocode](https://github.com/iamslash/learntocode/tree/master/leetcode/HighestGradeForEachStudent/a.sql)

# Quiz

* Multiple Apartments
* Open Requests
* Close All Requests
* Joins
* Denormalization
* Entity-Relationship Diagram
* Design Grade Database