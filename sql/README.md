# Intro

- 주로 사용하는 phrase들을 정리해보자.

# Material

* [use the idex luke](http://use-the-index-luke.com/)
* [sql @ w3schools](https://www.w3schools.com/sql/default.asp)

# Usage

## Select

```sql
SELECT CustomerName, City FROM Customers;
SELECT * FROM Customers;
```

## Select Distinct

```sql
SELECT DISTINCT Country FROM Customers;
SELECT COUNT(DISTINCT Country) FROM Customers;
SELECT Count(*) AS DistinctCountries
  FROM (SELECT DISTINCT Country FROM Customers);
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
SELECT CustomerID as ID, CustomerName AS Customer
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

# References

- [sqlzoo](http://sqlzoo.net/)
- [sql snippet](https://en.wikibooks.org/wiki/Structured_Query_Language/Snippets)
- [sakila](https://dev.mysql.com/doc/sakila/en/sakila-preface.html)
  - 비디오 대여점을 모델링한 example db이다. sql을 참고해서 공부하자.
