- [Abstract](#abstract)
- [Materials](#materials)
- [Fundamentals](#fundamentals)
  - [Index](#index)
  - [Pages](#pages)
  - [Clustered Index vs Non-clustered Index](#clustered-index-vs-non-clustered-index)
- [Index Access Principles](#index-access-principles)
  - [Principle 1: Fast Lookup](#principle-1-fast-lookup)
  - [Principle 2: Scan in One Direction](#principle-2-scan-in-one-direction)
  - [Principle 3: From Left To Right](#principle-3-from-left-to-right)
  - [Principle 4: Scan On Range Conditions](#principle-4-scan-on-range-conditions)
- [Index Supported Operations](#index-supported-operations)
  - [Inequality (!=)](#inequality-)
  - [Nullable Values (IS NULL and IS NOT NULL)](#nullable-values-is-null-and-is-not-null)
  - [Pattern Matching (LIKE)](#pattern-matching-like)
  - [Sorting Values (ORDER BY)](#sorting-values-order-by)
  - [Aggregating Values (DISTINCT and GROUP BY)](#aggregating-values-distinct-and-group-by)
  - [Joins](#joins)
  - [Subqueries](#subqueries)
  - [Data Manipulation (UPDATE and DELETE)](#data-manipulation-update-and-delete)
- [Why isn't Database Using My Index?](#why-isnt-database-using-my-index)
  - [The Index Can’t Be Used](#the-index-cant-be-used)
  - [No Index Will Be the Fastest](#no-index-will-be-the-fastest)
  - [Another index is faster](#another-index-is-faster)
- [Pitfalls and Tips](#pitfalls-and-tips)
  - [Indexes on Functions](#indexes-on-functions)
  - [Boolean Flags](#boolean-flags)
  - [Transforming Range Conditions](#transforming-range-conditions)
  - [Leading Wildcard Search](#leading-wildcard-search)
  - [Type Juggling](#type-juggling)
  - [Index-Only Queries](#index-only-queries)
  - [Filtering and Sorting With Joins](#filtering-and-sorting-with-joins)
  - [Exceeding the Maximum Index Size](#exceeding-the-maximum-index-size)
  - [JSON Objects and Arrays](#json-objects-and-arrays)
  - [Unique Indexes and Null](#unique-indexes-and-null)
  - [Location-Based Searching With Bounding-Boxes](#location-based-searching-with-bounding-boxes)

----

# Abstract

Database index. Hands on with [MySQL](/mysql/README.md).

# Materials

[Indexing Beyond the Basics](https://sqlfordevs.com/ebooks/indexing)

# Fundamentals

## Index

> * [SQL Unplugged 2013] 쉽고 재미있는 인덱스 이야기/ 씨퀄로 이장래](https://www.youtube.com/watch?v=TRfVeco4wZM)
> * [RDB index 정리](https://hyungjoon6876.github.io/jlog/2018/07/18/rdb-indexing.html)
> * [The chidb File Format](http://chi.cs.uchicago.edu/chidb/fileformat.html)

----

![](img/physlog.png)

위의 그림을 잘 살펴보자. xdb file 은 여러개의 page 로 구성되어 있다. 하나의 page
는 하나의 BTreeNode 를 저장한다. 하나의 BTreeNode 는 하나 이상의 record 를
저장한다. 또한 하나의 page 는 여러개의 cell 로 이루어져 있다. cell 은 record 에
대응된다. 

Courses 테이블은 schema 가 `CREATE TABLE Courses(Id INTEGER PRIMARY KEY, Name TEXT, Instructor INTEGER, Dept INTEGER)` 라고 해보자. primary key 인 id 를 hashing 한 값을 key 로 BTree 를 제작한다. 당연히 Id 를 조건으로 검색하면 빠르다. 

예를 들어 위 그림에서 id 를 hashing 한 값 `86` 을 검색하기 위해 BTree 를
`42->85->86` 순으로 검색에 성공했다.

그러나 Dept 를 조건으로 검색하면 느리다. Dept 의 인덱스를 제작한다. 즉, Dept 의
hashing 값을 key 로 B+Tree 를 제작한다. 당연히 Dept 를 조건으로 검색하면
빨라진다.

예를 들어 Dept 를 hashing 한 값 `42` 를 검색하기 위해 B+Tree 를 `10->42` 순으로
검색에 성공했다. `42` 는 primary key 인 ID 를 hashing 한 값 `67` 를 소유한 cell
을 가리키고 있다.

## Pages

* [How does SQLite work? Part 2: btrees! (or: disk seeks are slow don't do them!)](https://jvns.ca/blog/2014/10/02/how-does-sqlite-work-part-2-btrees/)
  * sqlite 의 database file 이 왜 pages 로 구성되어 있는지 설명한다.

----

database file 은 물리적으로 여러개의 pages 로 구성되야 한다.

CPU 는 disk 에서 데이터를 읽어들여 memory 로 로드해야 한다. 이때 데이터의 크기는 내가 찾고자 하는
것에 가까운 최소의 크기여야 한다. 내가 찾고자하는 데이터는 16 Byte 인데 1 MB 를 메모리에 로드하는 것은
낭비이다. 

그리고 filesystem 의 block size 는 4 KB 이다. 따라서 한번에 1 MB 를 메모리에서 읽는 것은 불가능하다.
여러번에 걸쳐 disk access 를 해야 한다. I/O 낭비가 발생한다.

따라서 데이터를 page 에 저장하고 그것의 크기는 block size 보다는 작고 
너무 작지 않게 설정하여 L1, L2 Cache hit 가 이루어 지도록 해야 한다.

참고로 chidb, sqlite 는 page 의 size 가 `1K` 이다.

## Clustered Index vs Non-clustered Index

> * [Difference between Clustered and Non-clustered index](https://www.geeksforgeeks.org/difference-between-clustered-and-non-clustered-index/)
> * [[MySQL] 인덱스구조 : 클러스터링인덱스/넌 클러스터링인덱스](https://pearlluck.tistory.com/m/54)

----

![](img/2022-10-23-07-57-59.png)

**Clustered Index** 는 Leaf Node 에 모든 Attribute Values 가 있다. SELECT 할 때
한번에 읽어 온다. 반면에 **Non-Clustered Index** 는 Leaf Node 가 **Clustered
Index** 의 주소를 갖는다. 다시한번 **Clustered Index** 를 찾아갈 필요가 있다. 추가적인 I/O 가 필요하다.

Index of `Roll_no is` is a **Clustered Index**.

```sql
create table Student
( Roll_No int primary key, 
Name varchar(50), 
Gender varchar(30), 
Mob_No bigint );

insert into Student
values (4, 'ankita', 'female', 9876543210 );

insert into Student 
values (3, 'anita', 'female', 9675432890 );

insert into Student 
values (5, 'mahima', 'female', 8976453201 ); 
```

There is no additional index data for `Roll_no`.

| Roll_No | Name | 	Gender | 	Mob_No |
|---|---|---|---|
| 3 | 	anita |	female	| 9675432890 |
| 4 |	ankita	| female |	9876543210 |
| 5 |	mahima |	female |	8976453201 |

Index of `Name` is a **Non-Clustered Index**.

```sql
create table Student
( Roll_No int primary key, 
Name varchar(50), 
Gender varchar(30), 
Mob_No bigint );

insert into Student 
values (4, 'afzal', 'male', 9876543210 );

insert into Student 
values (3, 'sudhir', 'male', 9675432890 );

insert into Student 
values (5, 'zoya', 'female', 8976453201 );

create nonclustered index NIX_FTE_Name
on Student (Name ASC); 
```

There is additional index data for `Name`. This needs record lookup.

| Name |	Row address |
|---|----|
| Afzal |	3452 |
| Sudhir |	5643 |
| zoya |	9876 |

일반적으로 테이블에 여러 개의 non-clustered index가 있을 때, clustered index를
auto-increment로 생성하는 것이 좋습니다. 그 이유는 다음과 같습니다.

* 인덱스 크기: auto-increment를 사용하여 생성한 clustered index는 숫자로 이루어져
있어 인덱스 크기가 작습니다. 이는 I/O 효율과 디스크 공간 절약에 도움이 됩니다.
* 삽입 성능: auto-increment 기반의 인덱스는 새 데이터가 인덱스 페이지 끝에
추가되기 때문에 페이지 분할 및 재구성이 최소화됩니다. 이로 인해 인덱스 생성과
유지에 관련된 오버헤드가 줄어들어 삽입 성능이 좋아집니다.
* 단순성: auto-increment로 생성된 clustered index는 단일 컬럼에 기반하기 때문에
인덱스 관리가 비교적 쉽습니다.

# Index Access Principles

## Principle 1: Fast Lookup

The primary purpose of an index is to provide a fast and efficient lookup
mechanism to retrieve the requested information from the database. Indexes can
significantly speed up the query execution process. Here is an example:

Consider a table employee with columns `id`, `name`, and `position`. Without an
index on the id column, a simple SQL query like the one below would require a
full table scan to find the desired row(s):

```sql
SELECT name, position FROM employee WHERE id = 100;
```

By creating an index on the `id` column, the database can efficiently look up the
location wherein rows matching the given `id` value can be found:

```sql
CREATE INDEX idx_employee_id ON employee (id);
```

Now, running the same SQL query would speed up the search process as the
database would use the index to quickly locate the desired row(s).

## Principle 2: Scan in One Direction

Indexes are often stored in a sorted order (e.g., B-trees) to allow a
one-directional scan when searching for specific values or ranges. This can
optimize the data retrieval process. Here's an example:

```sql
SELECT name, position FROM employee WHERE id > 50 ORDER BY id;
```

By utilizing a sorted index on the `id` column, the database can quickly locate
the first matching row `(id = 51)` and then scan in a single direction (forward)
through the index to find and return all matching rows.

## Principle 3: From Left To Right

**Indexes Are Used From Left to Right**

When creating a composite index on multiple columns, it's essential to
understand that the database scans the index from left to right, starting with
the most significant (leftmost) column.

For example, consider a table orders with columns customer_id, status, and
order_date. If we create an index on `(customer_id, status, order_date)`, it would
look like this:

```sql
CREATE INDEX idx_orders_customer_status_date ON orders (customer_id, status, order_date);
```

A query that filters by `customer_id` first, then the `status`, and finally the
`order_date`, will utilize the index more efficiently:

```sql
SELECT * 
  FROM orders 
 WHERE customer_id = 100 AND 
       status = 'shipped' AND 
       order_date > '2021-01-01';
```

**The Ordering Is Important**

The order of the columns in the composite index is important to optimize query
performance. 

For example, consider the table orders and queries with the following filtering
criteria:

Query A:

```sql
SELECT * FROM orders WHERE customer_id = 100 AND status = 'shipped';
```

Query B:

```sql
SELECT * FROM orders WHERE status = 'shipped' AND order_date > '2021-01-01';
```

Creating the index `(customer_id, status, order_date)` may be optimal for Query
A but not for Query B. For Query B, the optimal index would be 
`(status, order_date)`.

```sql
CREATE INDEX idx_orders_status_date ON orders (status, order_date);
```

**Skipping a Column**

If a query filters data from a middle column of a composite index, the remaining
rightmost columns may not be used efficiently.

For example, consider the index `(customer_id, status, order_date)` for the orders
table. A query that filters only by `status` may not use the index efficiently
since the leftmost column `(customer_id)` is not specified.

```sql
SELECT * FROM orders WHERE status = 'shipped' AND order_date > '2021-01-01';
```

In this case, it would be better to create a separate index on the status and
order_date columns.

**Overlapping Indexes**

Overlapping indexes may lead to redundancy and can unnecessarily increase
storage and maintenance costs. If an index covers another index entirely, the
larger index may be redundant.

For example, consider the following indexes on the orders table:

```sql
CREATE INDEX idx_orders_customer_status ON orders (customer_id, status);
CREATE INDEX idx_orders_customer_status_date ON orders (customer_id, status, order_date);
```

The index `(customer_id, status)` is entirely covered by the larger index
`(customer_id, status, order_date)`. In most cases, it would be beneficial to
remove the smaller index to improve storage and maintenance efficiency:

```sql
DROP INDEX idx_orders_customer_status;
```

## Principle 4: Scan On Range Conditions

The rangeable column should be last.

```sql
select * 
  from employee 
 WHERE country = 'US' AND married = 'yes' AND age > 28
```

As the query includes equality conditions (exact match) on the `country` and
`married` columns and a range condition on the `age` column, the better index
would be `(country, married, age)` instead of `(country, age, married)`:

```sql
CREATE INDEX idx_employee_country_married_age ON employee (country, married, age);
```

With this index, the database can first filter the rows by `country` and `married`
using exact matches and then efficiently process the range condition on `age`. By
placing the columns with equality conditions before the column with a range
condition, the index can be utilized more efficiently.

The SQL query would look like this:

```sql
SELECT * 
  FROM employee 
 WHERE country = 'US' AND married = 'yes' AND age > 28;
```

# Index Supported Operations

## Inequality (!=)

Inequality conditions (using the `!=` or `<>` operator) can also negatively
impact index usage and performance in some cases. The reason is that inequality
predicates do not efficiently narrow down the search space, as the database has
to scan more entries in the index to identify rows of interest.

For example, consider a table called products with columns `id`, `name`, `category`,
and `price`. We have an index on the `category` column:

```sql
CREATE INDEX idx_products_category ON products (category);
```

Now, let's look at the following query that uses an equality condition:

```sql
SELECT * FROM products WHERE category = 'Electronics';
```

With the given index, the database can quickly filter down and identify rows
having the exact category value of 'Electronics'.

Now, let's consider an inequality condition:

```sql
SELECT * FROM products WHERE category != 'Electronics';
```

In this case, the database needs to scan more entries in the index, as it has to
find all rows with a `category` value different from 'Electronics'. While the
index can still help limit the result set, it is less efficient than in the case
of equality conditions. Depending on the data distribution and the selectivity
of the inequality condition, the database might decide to perform a full table
scan, which will negatively impact the performance.

Keep in mind that the performance impact can vary depending on the database
management system being used, the inequality condition's selectivity, and the
index's overall effectiveness in the specific scenario.

## Nullable Values (IS NULL and IS NOT NULL)

Nullable values refer to columns in a database table that have the potential to
contain a `NULL` value. In SQL, `NULL` is unique in that it represents the absence
of a value or an unknown value. This distinct characteristic requires a
different approach when handling `NULL` values in SQL queries. To specifically
check for `NULL` or not `NULL` values, you must use the `IS NULL` and `IS NOT NULL`
conditions, respectively.

Suppose there is an employees table with the following columns:

- id (integer)
- name (varchar)
- supervisor_id (integer, nullable)

Here, the `supervisor_id` column has nullable values, meaning some employees might
not have a supervisor.

Example 1: Using the IS NULL condition

Let's search for all employees with no supervisor:

```sql
SELECT * 
  FROM employees
 WHERE supervisor_id IS NULL;
```

This query returns all rows in the employees table where the `supervisor_id`
column has a NULL value.

Example 2: Using the IS NOT NULL condition

Now, let's fetch all employees who do have a supervisor:

```sql
SELECT * 
  FROM employees
 WHERE supervisor_id IS NOT NULL;
```

This query returns all rows in the employees table where the `supervisor_id`
column has a non-NULL value.

When working with nullable columns, creating an index can help optimize query
performance. For example, you could create an index on the `supervisor_id` and
name columns:

```sql
CREATE INDEX idx_employees_supervisor_name ON employees(supervisor_id, name);
```

This index will be efficient for queries that involve both the `supervisor_id` and
name columns. However, remember that `IS NOT NULL` conditions may not always make
efficient use of indexes, so you should consider the specific queries and
indexes your application uses.

In conclusion, handling nullable values in SQL requires the use of `IS NULL` and
`IS NOT NULL` conditions to properly deal with `NULL` values. Creating the right
indexes helps improve query performance and ensures the efficiency of your SQL
operations.

## Pattern Matching (LIKE)

The use of wildcards with pattern matching (LIKE) in SQL queries can be quite
helpful when searching for strings with similar characters or patterns. the use
of wildcards can be internally rewritten as range conditions by the database
system for better optimization.

Consider a table called contacts with columns `id`, `type`, `firstname`, and
`lastname`. We have a composite index on the type and `firstname` columns:

```sql
CREATE INDEX idx_contacts_type_firstname ON contacts (type, firstname);
```

Now, let's look at an example of a pattern-matching query using wildcards:

```sql
SELECT * FROM contacts WHERE type = 'customer' AND firstname LIKE 'Tobi%';
```

Here, we are looking for all customer rows whose `firstname` starts with "Tobi".
The composite index can help optimize this query since the equality condition
`type = 'customer'` comes before the wildcard search `firstname LIKE 'Tobi%'`.
This order of columns is important for efficient index utilization.

Internally, the database can rewrite the pattern-matching condition as a range
condition:

```sql
firstname >= 'Tobi' AND firstname < 'Tobj'
```

This will enable the database to use the index by first filtering rows with 
`type = 'customer'` and then perform a directional scan for matching firstname values
within the specified range.

Remember that wildcards can have an impact on performance when used
inappropriately. For instance, if a wildcard character is used at the beginning
of the search pattern, such as `%Tobi`, the index might not be utilized
efficiently or at all, potentially leading to a full table scan. It's important
to consider the location of the wildcards and the distribution of values in the
indexed columns to ensure optimal indexing and query performance.

## Sorting Values (ORDER BY)

Sorting values in a database is a crucial operation to organize and display the
data according to specific requirements. The SQL ORDER BY clause is used to sort
data in either ascending (ASC) or descending (DESC) order based on one or more
columns.

Consider a sample table issues with the following columns:

- id (integer)
- type (varchar)
- severity (integer)
- comments_num (integer)

To filter out all the 'new' issues and display them ordered by the highest
severity and the number of comments, you would use the following SQL query:

```sql
SELECT * 
  FROM issues
 WHERE type = 'new'
 ORDER BY severity DESC, comments_num DESC;
```

In this example, the `ORDER BY` clause sorts the result set first by the
`severity` column in descending order, and then by the `comments_num` column,
also in descending order. This way, you will see the highest `severity` issues
with the most comments at the top of the result set.

Creating an index on the sorting columns will allow the database to fetch rows
from the index more efficiently. To create an index on the `severity` and
`comments_num` columns, you can use the following SQL statement:

```sql
CREATE INDEX idx_issues_severity_comments ON issues(severity DESC, comments_num DESC);
```

This index will avoid additional sorting steps, thus improving the performance
of your query, especially when dealing with large datasets.

If you work with MySQL, you can tweak the `sort_buffer_size` configuration to
adjust the memory threshold for in-memory sorting. In PostgreSQL, this can be
achieved with the `work_mem` setting.

In summary, the SQL `ORDER BY` clause is an essential tool to sort and organize
data within your database. Creating an index on sorting columns helps improve
performance and avoid unnecessary memory usage. Adjusting memory settings like
`sort_buffer_size` or `work_mem` can also aid in optimizing sorting operations.

## Aggregating Values (DISTINCT and GROUP BY)

These operations often aggregate a large number of rows, and optimizing them
properly can have a significant impact on query performance.

The `DISTINCT` keyword is used to return a result set with unique values for a
specific column. The `GROUP BY` clause is used to group rows that have the same
values in specified columns into groups, allowing aggregation functions like
`COUNT`, `AVG`, etc., to be applied. They are often used together with `WHERE`
conditions, affecting the order in which columns must appear in an index for
optimal performance.

The following examples illustrate query optimization using `DISTINCT`, `GROUP BY`,
and `WHERE` conditions:

- Basic DISTINCT example:

```sql
SELECT DISTINCT country FROM users;
```

Equivalent GROUP BY example:

```sql
SELECT country FROM users GROUP BY country;
```

- Simple GROUP BY:

```sql
SELECT is_paying, COUNT(*)
  FROM users
 GROUP BY is_paying;
```

- GROUP BY with multiple columns:

```sql
SELECT is_paying, gender, COUNT(*)
  FROM users
 GROUP BY is_paying, gender;
```  

- GROUP BY with a WHERE condition:

```sql
SELECT is_paying, gender, COUNT(*)
  FROM users
 WHERE onboarding = 'yes'
 GROUP BY is_paying, gender;
```

As explained in the passage, the columns used in the WHERE part must always be
added before the columns in the GROUP BY clause. This ensures the database can
efficiently filter rows and perform the grouping.

- GROUP BY with a range condition in the WHERE clause:

```sql
SELECT is_paying, gender, COUNT(*)
  FROM users
 WHERE age BETWEEN 20 AND 29
 GROUP BY is_paying, gender;
```

In this case, the index may not be efficient for the GROUP BY operation, and a
temporary mapping table may be required.

- GROUP BY with aggregate functions:

```sql
SELECT is_paying, gender, AVG(projects_cnt)
  FROM users
 GROUP BY is_paying, gender;
```

To optimize for aggregate functions, the columns used in the `SELECT` clause
should be included in the index after the filtering and grouping columns.

In conclusion, when using aggregating values like `DISTINCT` and `GROUP` BY in
SQL queries, properly optimizing indexes is crucial to ensure efficient query
performance, especially when aggregating large numbers of rows.

## Joins

Joins are a critical aspect of SQL when querying data from multiple tables. The
major challenge with joins is to optimize their performance. To do this, it is
essential to understand how databases execute these joins and which indexes are
needed.

Databases typically execute joins using a method called "nested-loop join,"
which is similar to a for-each or for-in loop in a programming language. One
table is accessed with all filters applied, and the matching rows serve as the
iteration data for the loop. For every one of these rows, another query on a
different table is executed using the values from the first table.

For example, consider the following join query:

```sql
SELECT employee.*
  FROM employee
  JOIN department USING(department_id)
 WHERE employee.salary > 100000 AND department.country = 'NR';
```

This query can be de-constructed into two independent queries:

```sql
SELECT *
FROM employee
WHERE salary > 100000;

-- for each matching row from employee:
SELECT *
  FROM department
 WHERE country = 'NR' AND 
       department_id = :value_from_employee_table;
```

By breaking it down into separate queries, you can now create and optimize
indexes on the `employee` and `department` tables using your existing knowledge. In
this case, the order of columns doesn't matter since they are both equality
checks.

The strategy for optimizing a two-table join can also be applied when using
joins with more tables. The database will simply use more nested loops for each
additional table.

It's vital to note that the join-order in SQL queries is not fixed. SQL is a
declarative language specifying what data you want, but not how to retrieve it.
The database optimizer is responsible for finding the fastest method to execute
the query.

For instance, considering the sample data provided, it may be more efficient to
first search for departments in Nauru and then find employees earning more than
`$100,000/year` in those departments, like this:

```sql
SELECT *
  FROM department
 WHERE country = 'NR';

-- for each matching row from department:
SELECT *
  FROM employee
 WHERE salary > 100000 AND department_id = :value_from_department_table;
```

This reduces the number of operations since only two queries are executed within
the loop compared to `511` with the original approach.

Always remember that the order in which you write joined tables doesn't
determine the order in which they are executed. A different execution order can
significantly improve query performance. The database will estimate the fastest
approach, but it's crucial to provide the necessary indexes to allow it to make
that choice. So always add all the indexes required for the database to execute
joins in any possible order. If you miss an essential index, the database may
not use the fastest join order.

## Subqueries

Subqueries are often misunderstood as being slow, but the actual issue is
usually the lack of appropriate indexes. To create suitable indexes for
subqueries, you should optimize each subquery independently. The key is to
understand the difference between independent and dependent subqueries because
both types have different index requirements.

Independent Subqueries:

Consider the following query with an independent subquery:

```sql
SELECT *
  FROM products
 WHERE remaining > 500 AND category_id = (
    SELECT category_id
      FROM categories
     WHERE type = 'book' AND name = 'Science fiction'
)
```

The subquery is independent because no tables from the outer query are used
within it. It is executed only once and the result is used in the outer query's
condition. To create an index for the subquery, you can ignore its context
within the more extensive query:

- Create an index on the `categories` table using the `type` and `name` columns.
- Replace the subquery in the SQL statement with the computed category_id value.
- Create a final index for the products table using the `category_id` and
  remaining columns.

Dependent Subqueries:

Now, consider a query with a dependent subquery:

```sql
SELECT *
  FROM products
 WHERE remaining = 0 AND EXISTS (
    SELECT *
      FROM sales
     WHERE created_at >= '2023-01-01' AND product_id = products.product_id
)
```

The subquery is dependent because it references a table from the outer query
(i.e., `products.product_id`). To create indexes for dependent subqueries:

- Execute the outer query first, creating an index on the products table for the
  remaining column.
- For each matching products row, execute the subquery with a different value
  for `products.product_id`. Create an index for the `sales` table using the
  `product_id` and `created_at` columns.

In this case, the outer query's `EXISTS` condition is satisfied as soon as the
subquery finds one matching row. You can think of it as automatically applying a
`LIMIT 1` to the subquery. However, you still need to build the index according to
range condition principles (Index Access Principle 4: Scan On Range Conditions)
for optimal efficiency.

In conclusion, when working with subqueries in SQL, it is crucial to understand
the difference between independent and dependent subqueries and create
appropriate indexes for each type. This will improve the performance of your
queries and ensure optimal execution.

## Data Manipulation (UPDATE and DELETE)

Data manipulation refers to the process of modifying or deleting data within a
database using SQL queries. The two primary SQL statements used for data
manipulation are UPDATE and `DELETE`. Although these statements are known for
slower execution compared to `SELECT`, it's important to note that their
performance can be optimized, especially the part that searches for matching
rows.

Imagine you have a `products` table with the following columns:

- id (integer)
- name (varchar)
- price (integer)
- status (varchar)

Here are some examples of `UPDATE` and `DELETE` queries with corresponding `SELECT`
queries that can be optimized:

Example 1: Data Modification (UPDATE)

Suppose you need to update the prices of all the items with the status 'Sale'.

`UPDATE` Query:

```sql
UPDATE products SET price = price * 0.9 WHERE status = 'Sale';
```

Converted `SELECT` Query:

```sql
SELECT * FROM products WHERE status = 'Sale';
```

The `SELECT` query can be optimized by creating an index based on the status
column, which can also be used to speed up the `UPDATE` statement:

```sql
CREATE INDEX idx_products_status ON products(status);
```

Example 2: Data Deletion (DELETE)

Now let's say you want to delete all the products with a price above 1000.

DELETE Query:

```sql
DELETE FROM products WHERE price > 1000;
```

Converted SELECT Query:

```sql
SELECT * FROM products WHERE price > 1000;
```

The SELECT query can be optimized by creating an index on the price column,
which can also help speed up the `DELETE` statement:

```sql
CREATE INDEX idx_products_price ON products(price);
```

In conclusion, data manipulation using `UPDATE` and `DELETE` queries can be
optimized by considering the way they find matching rows. By rewriting these
queries as `SELECT` statements, you can analyze and optimize them using
appropriate indexes, thereby improving their overall performance.

# Why isn't Database Using My Index?

Understanding query execution is crucial for determining why an index might be
ignored. Each query undergoes a three-step process:

- Parsing: The query is broken down into components to identify its intention.
- Simple Plan Creation: An initial query plan is developed, including full-table
  scans at this stage, ensuring there is always a viable execution method.
- Optimization: Indexes are evaluated, and other optimizations are considered to
  improve the query plan. If no better optimization is found, the initial plan
  is retained.

## The Index Can’t Be Used

There 4 cases including **Column Transformations**, **Incompatible Column Ordering**,
**Operations Not Supported by the Index**, **The Index Is Invisible**.

**Column Transformations**

When performing transformations or applying functions to a column in a query,
the index on that column might not be usable.

```sql
-- as-is
-- Suppose we have an index on the 'salary' column:
CREATE INDEX salary_idx ON employees(salary);

-- The index may not be used when there is a transformation 
-- applied to the column in the WHERE clause:
SELECT * FROM employees WHERE SQRT(salary) > 2000;

-- to-be
CREATE INDEX salary_transformed_idx ON employees(SQRT(salary));
```

**Incompatible Column Ordering**

The order of columns in an index matters. If the query does not match the order
of columns in the index, the index might not be utilized.

```sql
-- Suppose we have an index on columns 'last_name' and 'first_name':
CREATE INDEX name_idx ON employees(last_name, first_name);
-- The index may not be used if the query filters only by 'first_name':

SELECT * FROM employees WHERE first_name = 'John';

-- In this case, consider creating an index with 'first_name' as the leading column:
CREATE INDEX first_name_idx ON employees(first_name);
```

**Operations Not Supported by the Index**

Indexes may not support certain operations or cannot be applied to specific data
types. For example, the standard B-tree index does not support array operations
or full-text searching.

```sql
-- Using PostgreSQL tsvector data type (text search):
-- Creating a tsvector column and an index on that column
ALTER TABLE articles ADD COLUMN tsv tsvector;
CREATE INDEX articles_tsv_idx ON articles USING gin(tsv);

-- A query that requires full-text search
SELECT * FROM articles WHERE tsv @@ to_tsquery('english', 'search_term');
-- In this case, we used a special GIN (Generalized Inverted Index) 
-- to support full-text search operations. Standard B-tree indexes 
-- would not work with tsvector data type and text search operations.
```

**The Index Is Invisible**

In some databases (such as PostgreSQL), you can create invisible indexes to test
their impact before making them public. An invisible index won't be used for
queries unless explicitly specified.

```sql
-- In PostgreSQL, create an invisible index:
CREATE INDEX employees_test_idx ON employees(salary) WHERE (salary > 5000) WITH (visible = false);
-- To use the invisible index, we need to specify it explicitly in the query:

SET enable_indexscan TO off;
SET enable_bitmapscan TO off;
SELECT * FROM employees WHERE salary > 5000;

-- By setting enable_indexscan and enable_bitmapscan to off, we ensure that 
-- the query will not use normal indexes. But it would still use the 
-- invisible index in the best cases, allowing us to measure its impact.
```

## No Index Will Be the Fastest

Index is not the silver bullet.

**Loading Many Rows**

As a rule of thumb, if you need to fetch more than 20-30% of a table's rows, a
full table scan might be more efficient than using an index. Keep in mind that
this percentage can vary depending on the specific characteristics of the
database, the table schema, and the hardware resources.

However, it is important to note that using a full table scan can be
resource-intensive, as it requires reading all the rows in the table. In
situations where only a small percentage of rows need to be fetched, indexes are
often more efficient because they help the database engine quickly locate and
access the relevant data, bypassing irrelevant rows.

**Small Tables**

When dealing with a small number of records, an index can still be quite useful
in optimizing data retrieval performance, especially when the queries involve
filtering or searching on specific columns.

Even if a table has a small number of rows, using an index allows the database
engine to quickly locate the relevant data by referring to the index data
structure rather than scanning the entire table. This can help speed up the
query execution and reduce system resource utilization, particularly when the
table has many columns, some of which are not needed for the query at hand.

However, there might be certain situations where indexing a small table might
not yield significant performance improvements. Indexes come with some overhead
in terms of disk space and maintenance, such as updating the index when new
records are inserted or deleted, and rebuilding or reorganizing index
structures. Due to this overhead, adding an index to a very small table could
sometimes have minimal impact on query performance.

It is important to monitor and analyze the database's query performance and
assess whether indexing specific columns in small tables yields the desired
performance improvements. If indexing doesn't bring significant benefits, you
may decide not to create an index for that particular small table.

**Outdated Statistics**

Outdated statistics might occur due to several reasons, including:

- Large amounts of new data being inserted, deleted, or updated in the table.
- The auto-update mechanism for statistics (if present) is not triggered yet or
  configured with improper thresholds.
- Periodic update jobs for statistics have not been scheduled, or existing jobs
  have failed to execute.

To address the issue of outdated statistics, you should consider taking the
following steps:

- Regularly update the statistics, especially after significant data
  modifications in the table. You can use the database-specific commands for
  updating statistics, such as `ANALYZE TABLE` in **MySQL** or `ANALYZE` in
  **PostgreSQL**.
- If the automatic statistics updating feature is supported and enabled in your
  database, ensure that the auto-update thresholds are set appropriately
  according to the data change frequency and the specific needs of your database
  workload.
- Schedule periodic jobs to update statistics, especially for large or
  frequently modified tables, to ensure that the query optimizer always has the
  most recent and accurate information.

Keeping the statistics up-to-date helps the query optimizer make better
decisions when choosing execution plans and optimizing the use of indexes, which
in turn results in improved query performance and a more efficient utilization
of system resources.

## Another index is faster

**Multiple Conditions**

Here are some SQL examples demonstrating the scenarios mentioned earlier, where
a multi-column index can be faster and more efficient. In these examples, we'll
be using a sample orders table with three columns: `order_id`, `customer_id`,
and `order_date`.

- Creating a multi-column index:

```sql
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);
```

This index combines `customer_id` and `order_date`, which can provide better
filtering capabilities.

- Query example using multi-column index for improved filtering:

Suppose you want to retrieve all orders for a specific customer within a certain
date range. A multi-column index can efficiently filter both conditions at once:

```sql
SELECT * FROM orders 
WHERE customer_id = 42 AND order_date BETWEEN '2021-01-01' AND '2021-12-31';
```

In this case, the multi-column index `idx_orders_customer_date` can be utilized to
quickly locate the relevant rows that match both conditions.

- Query example using multi-column index for efficient sorting:

When you need to retrieve orders for a specific customer and sort them by date:

```sql
SELECT * 
  FROM orders 
 WHERE customer_id = 42
 ORDER BY order_date DESC;
```

The multi-column index `idx_orders_customer_date` can help speed up the sorting
process, as the index already has the order_date sorted for each `customer_id`.

- Query example of an index-covered query:
  
If you only need to fetch specific columns included in the index, such as the
customer_id and order_date of all orders:

```sql
SELECT customer_id, order_date
  FROM orders;
```

In this case, the database engine can use the `idx_orders_customer_date` index to
retrieve the required data without accessing the actual table data, resulting in
an index-only scan and faster query execution.

Keep in mind that the effectiveness of the multi-column index depends on the
specific data distribution, query patterns, and the order of columns in the
index. Analyzing your database schema and workload will help you determine the
optimal index structure for faster and more efficient queries.

**Join**

Here are some SQL examples demonstrating the use of indexes in optimizing join
operations. In these examples, we'll use two sample tables: `orders` and
`customers`. The `orders` table has the columns: `order_id`, `customer_id`, and
`order_date`, while the `customers` table has the columns: `customer_id`,
`name`, and `email`.

- Creating an index on the customer_id columns in both tables:

```sql
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_customers_customer_id ON customers (customer_id);
```

These indexes will enable faster search and retrieval of matching rows during
join operations involving the `customer_id` column.

- Query example using **INNER JOIN** with a proper index:

Suppose you want to retrieve the list of orders along with customer information.
You can use an **INNER JOIN** for this purpose:

```sql
SELECT o.order_id, o.order_date, c.customer_id, c.name, c.email
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id;
```

With the indexes created on the `customer_id` column in both tables, the
database engine can efficiently use the indexed columns to locate the relevant
rows and perform the join operation much faster compared to a full table scan.

- Query example using **LEFT JOIN** with a proper index:

Imagine you want to retrieve a list of all customers, along with their order
information if available:

```sql
   SELECT c.customer_id, c.name, c.email, o.order_id, o.order_date
     FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;
```

In this case, the indexes on the `customer_id` column can help the database
engine efficiently perform the **LEFT JOIN**, with improved row retrieval and
reduced I/O operations.

Keep in mind that the performance improvement and efficiency of using indexes in
join operations depend on the specific data distribution, query patterns, and
the columns involved in the join condition. It's crucial to analyze your
database schema and workload to determine the optimal index structure and
indexing strategies to ensure faster and more efficient join operations.

**Ordering**

Using an index with the appropriate ordering is important for performance
optimization when executing queries with ordering conditions, such as in your
example:

```sql
... WHERE type = 'open' ORDER BY date DESC LIMIT 10
```

To leverage the index for sorting in your example, you should create an index on
the columns included in your query (type and date), such as:

```sql
CREATE INDEX idx_orders_type_date ON orders (type, date DESC);
```

This index will help the database engine to efficiently filter rows by the
'open' type and quickly locate the 10 most recent rows without additional
sorting.

# Pitfalls and Tips

## Indexes on Functions

**Functional Indexes**

```sql
CREATE TABLE contacts (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    birthday datetime NOT NULL
);
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());

-- Full scan
EXPLAIN SELECT * FROM contacts;
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    3 |   100.00 | NULL  |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------+

-- Index scan
EXPLAIN SELECT * FROM contacts WHERE id = 1;
+----+-------------+----------+------------+-------+---------------+---------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type  | possible_keys | key     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+-------+---------------+---------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | const | PRIMARY       | PRIMARY | 8       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+-------+---------------+---------+---------+-------+------+----------+-------+

-- Full scan
EXPLAIN SELECT * FROM contacts WHERE month(birthday) = 5;
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table    | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | contacts | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    3 |   100.00 | Using where |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------------+

-- Index scan
CREATE INDEX idx_contacts_birthmonth ON contacts ((month(birthday)));
EXPLAIN SELECT * FROM contacts WHERE month(birthday) = 5;
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys           | key                     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ref  | idx_contacts_birthmonth | idx_contacts_birthmonth | 5       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
```

**Vitual Column Indexes**

```sql
DROP TABLE contacts;
CREATE TABLE contacts (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    birthday datetime NOT NULL,
    birthday_month TINYINT AS (month(birthday)) VIRTUAL NOT NULL,
    INDEX idx_contacts_birthmonth (birthday_month)
);
DESC contacts;
+----------------+----------+------+-----+---------+-------------------+
| Field          | Type     | Null | Key | Default | Extra             |
+----------------+----------+------+-----+---------+-------------------+
| id             | bigint   | NO   | PRI | NULL    | auto_increment    |
| birthday       | datetime | NO   |     | NULL    |                   |
| birthday_month | tinyint  | NO   | MUL | NULL    | VIRTUAL GENERATED |
+----------------+----------+------+-----+---------+-------------------+
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());

-- Index scan
EXPLAIN SELECT * FROM contacts WHERE month(birthday) = 5;
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys           | key                     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ref  | idx_contacts_birthmonth | idx_contacts_birthmonth | 1       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+

-- Index scan
EXPLAIN SELECT * FROM contacts WHERE birthday_month = 5;
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys           | key                     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ref  | idx_contacts_birthmonth | idx_contacts_birthmonth | 1       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
```

## Boolean Flags

Boolean flags are not suitable for indexing in MySQL for several reasons:

* **Low cardinality**: Boolean flags have very low cardinality because they can
  only have two possible values: true or false (1 or 0). Indexes are typically
  more effective when the indexed column has a high number of distinct values. In
  the case of a Boolean flag, the index would end up pointing to many records with
  the same value, which decreases the index's efficiency and selectivity.
* **Poor query optimization**: Since Boolean flags have low cardinality, it
  makes it difficult for the query optimizer to make efficient execution plans.
  The optimizer may not choose to use an index on a Boolean flag due to its low
  selectivity, and even when it does, the full table scan might be faster than
  using the index, which defeats the purpose of having an index in the first
  place.
* **Larger index size**: Indexing a Boolean flag can increase the size of the
  index and add extra storage and maintenance overhead. Since the index may not
  be utilized effectively, the additional cost and complexity may not be
  worthwhile.
* **Limited performance improvement**: Using an index on a Boolean flag may not
  significantly improve query performance in scenarios where the distribution of
  values is highly skewed. For example, if 95% of records have the flag set to
  true and the query filters on the false value, the index may not provide
  substantial performance gains compared to a full table scan.

In conclusion, Boolean flags are not suitable for indexing in MySQL primarily
due to their low cardinality and limited effectiveness in query optimization. It
is generally more beneficial to index columns with higher cardinality and use
other query optimization techniques to improve performance for Boolean flag
columns.

## Transforming Range Conditions

Remove range conditions such as `WHERE stars > 1000` for better performance. You
might fetch unnecessary rows in the middle of fetching with the index because of
range conditions.

For solving range condition problems **virtual columns**, **functional indexes**
are good solutions.

**Index of (stars, language)**

```sql
CREATE TABLE trends (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    language VARCHAR(128) NOT NULL,
    stars INT NOT NULL,
    sponsors INT NOT NULL
);
show index from trends;
+--------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY  |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 20, 10);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 21, 11);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 22, 12);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 23, 13);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 24, 14);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 25, 15);
CREATE INDEX idx_stars_language ON trends (stars, language);
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name           | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY            |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            1 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            2 | language    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+


-- Full scan
explain select * from trends where language = 'TypeScript';
+----+-------------+--------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+--------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | trends | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    6 |    16.67 | Using where |
+----+-------------+--------+------------+------+---------------+------+---------+------+------+----------+-------------+

-- Index scan
explain select * from trends where stars > 20;
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys      | key                | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | trends | NULL       | range | idx_stars_language | idx_stars_language | 4       | NULL |    5 |   100.00 | Using index condition |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+

-- Index scan
explain select * from trends where language = 'TypeScript' AND stars > 20;
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys      | key                | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | trends | NULL       | range | idx_stars_language | idx_stars_language | 4       | NULL |    5 |    16.67 | Using index condition |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+

-- Index scan
explain select * from trends where stars > 20 AND language = 'TypeScript';
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys      | key                | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | trends | NULL       | range | idx_stars_language | idx_stars_language | 4       | NULL |    5 |    16.67 | Using index condition |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+

-- Index scan
explain analyze select * from trends where stars > 20 AND language = 'TypeScript';
-> Index range scan on trends using idx_stars_language over (20 < stars), with index condition: 
     ((trends.`language` = 'TypeScript') and (trends.stars > 20))  
       (cost=2.51 rows=5) (actual time=0.0513..0.0618 rows=1 loops=1)

-- Index scan (Query optimizer worked)
explain analyze select * from trends where language = 'TypeScript' AND stars > 20;
-> Index range scan on trends using idx_stars_language over (20 < stars), with index condition: 
     ((trends.`language` = 'TypeScript') and (trends.stars > 20))  
       (cost=2.51 rows=5) (actual time=0.0511..0.0675 rows=1 loops=1)
```

**Index of (language, stars)**

```sql
CREATE INDEX idx_language_stars ON trends (language, stars);

show index from trends;
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name           | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY            |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            1 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            2 | language    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars |            1 | language    | A         |           3 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars |            2 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

-- Index scan
explain select * from trends where language = 'TypeScript';
+----+-------------+--------+------------+------+--------------------+--------------------+---------+-------+------+----------+-------+
| id | select_type | table  | partitions | type | possible_keys      | key                | key_len | ref   | rows | filtered | Extra |
+----+-------------+--------+------------+------+--------------------+--------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | trends | NULL       | ref  | idx_language_stars | idx_language_stars | 514     | const |    2 |   100.00 | NULL  |
+----+-------------+--------+------------+------+--------------------+--------------------+---------+-------+------+----------+-------+
```

**Index of (language, stars, sponsors)**

```sql
CREATE INDEX idx_language_stars_sponsors ON trends (language, stars, sponsors);

show index from trends;
+--------+------------+-----------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name                    | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+-----------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY                     |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language          |            1 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language          |            2 | language    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars          |            1 | language    | A         |           3 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars          |            2 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars_sponsors |            1 | language    | A         |           3 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars_sponsors |            2 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars_sponsors |            3 | sponsors    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+-----------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

-- Index scan
explain select * from trends where language = 'TypeScript';
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys                                  | key                         | key_len | ref   | rows | filtered | Extra       |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-------------+
|  1 | SIMPLE      | trends | NULL       | ref  | idx_language_stars,idx_language_stars_sponsors | idx_language_stars_sponsors | 514     | const |    2 |   100.00 | Using index |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-------------+

-- Index scan
explain select * from trends where language = 'TypeScript' ORDER BY sponsors ASC;
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-----------------------------+
| id | select_type | table  | partitions | type | possible_keys                                  | key                         | key_len | ref   | rows | filtered | Extra                       |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-----------------------------+
|  1 | SIMPLE      | trends | NULL       | ref  | idx_language_stars,idx_language_stars_sponsors | idx_language_stars_sponsors | 514     | const |    2 |   100.00 | Using index; Using filesort |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-----------------------------+
```

**Virtual Columns (No need for range conditions)**

Index of (language, popular, sponsors) is better than range conditions. But
[PostgreSQL](/postgresql/README.md) doesn't support this.

```sql
CREATE TABLE trends (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    language VARCHAR(128) NOT NULL,
    stars INT NOT NULL,
    popular TINYINT AS(IF(stars > 20, 1, 0)) VIRTUAL NOT NULL,
    sponsors INT NOT NULL
);
CREATE INDEX idx_language_popular_sponsors on trends(language, popular, sponsors);
SHOW INDEX from trends
+--------+------------+-------------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name                      | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+-------------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY                       |            1 | id          | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_popular_sponsors |            1 | language    | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_popular_sponsors |            2 | popular     | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_popular_sponsors |            3 | sponsors    | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+-------------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 20, 10);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 21, 11);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 22, 12);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 23, 13);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 24, 14);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 25, 15);

explain analyze select * from trends where language = 'TypeScript' AND popular = 1 ORDER BY sponsors ASC;
-> Index lookup on trends using idx_language_popular_sponsors (language='TypeScript', popular=1)  
     (cost=0.35 rows=1) (actual time=0.0563..0.0614 rows=1 loops=1)
```

**Function Indexes (No need for range conditions)**

```sql
CREATE TABLE trends (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    language VARCHAR(128) NOT NULL,
    stars INT NOT NULL,
    sponsors INT NOT NULL
);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 20, 10);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 21, 11);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 22, 12);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 23, 13);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 24, 14);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 25, 15);
CREATE INDEX idx_language_popular_sponsors ON trends (language, (IF(stars > 20, 1, 0)), sponsors);

explain analyze 
select * 
  from trends 
 where language = 'TypeScript' AND 
       IF(stars > 20, 1, 0) = 1 
 ORDER BY sponsors ASC;
-> Index lookup on trends using idx_language_popular_sponsors (language='TypeScript', if((stars > 20),1,0)=1)   
     (cost=0.35 rows=1) (actual time=0.0497..0.0507 rows=1 loops=1)
```

## Leading Wildcard Search

Wildcard condition is converted to range condition. For example, `WHERE name
LIKE 'Tob%s'` is converted to `WHEREname >= 'Tob' AND name < 'Toc`. 

Remove leading wildcard for better performance. `WHERE name LIKE '%obs'` can
lead to poor performance When a leading wildcard is used, the database engine
cannot utilize the index efficiently since it has to search for all possible
combinations of characters preceding the search term (e.g., `{%Iobs, Aobs, Bobs,..., zobs, ...}` ). This results in a full table scan, which can be slow for
large tables.

But [PostgreSQL](/postgresql/README.md) supports indexing of wildcard any position.

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX trgm_idx ON contacts USING GIN (name gin_trgm_ops);
```

## Type Juggling

When we query `int` attributes for `varchar` column index, **full scan**
happens. But when we query `string` attributes for `int` column index, 
**index scan** happens.

The index `idx_orders` is not used for 
`EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718'`. 
Because `EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718'`
is changed to `SELECT * FROM orders WHERE CAST(payment_id AS UNSIGNED) = 57013925718`. 

```sql
CREATE TABLE orders (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    payment_id VARCHAR(255) NOT NULL
);
INSERT INTO orders(payment_id) VALUES (`57013925718`);
INSERT INTO orders(payment_id) VALUES (`57013925728`);
INSERT INTO orders(payment_id) VALUES (`57013925738`);
CREATE INDEX idx_orders ON orders (payment_id);

-- Index scan
EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718';
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys | key        | key_len | ref   | rows | filtered | Extra       |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
|  1 | SIMPLE      | orders | NULL       | ref  | idx_orders    | idx_orders | 1022    | const |    1 |   100.00 | Using index |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+

-- Full scan
EXPLAIN SELECT * FROM orders WHERE payment_id = 57013925718;
+----+-------------+--------+------------+-------+---------------+------------+---------+------+------+----------+--------------------------+
| id | select_type | table  | partitions | type  | possible_keys | key        | key_len | ref  | rows | filtered | Extra                    |
+----+-------------+--------+------------+-------+---------------+------------+---------+------+------+----------+--------------------------+
|  1 | SIMPLE      | orders | NULL       | index | idx_orders    | idx_orders | 1022    | NULL |    3 |    33.33 | Using where; Using index |
+----+-------------+--------+------------+-------+---------------+------------+---------+------+------+----------+--------------------------+
```

The opposite case works.

```sql
CREATE TABLE orders (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    payment_id bigint NOT NULL
);
INSERT INTO orders(payment_id) VALUES (57013925718);
INSERT INTO orders(payment_id) VALUES (57013925728);
INSERT INTO orders(payment_id) VALUES (57013925738);
CREATE INDEX idx_orders ON orders (payment_id);

-- Index Scan
EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718';
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+--------------------------+
| id | select_type | table  | partitions | type | possible_keys | key        | key_len | ref   | rows | filtered | Extra                    |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+--------------------------+
|  1 | SIMPLE      | orders | NULL       | ref  | idx_orders    | idx_orders | 8       | const |    1 |   100.00 | Using where; Using index |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+--------------------------+

-- Index Scan
EXPLAIN SELECT * FROM orders WHERE payment_id = 57013925718;
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys | key        | key_len | ref   | rows | filtered | Extra       |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
|  1 | SIMPLE      | orders | NULL       | ref  | idx_orders    | idx_orders | 8       | const |    1 |   100.00 | Using index |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
```

## Index-Only Queries

Index-only queries, also known as "covering indexes" or "covered queries," are a
type of query in MySQL wherein all the data required to satisfy the query can be
retrieved from the index itself, without accessing the actual table data (rows).

When a query uses a covering index, the MySQL query optimizer can avoid reading
the rows from the table, making the query execution faster and more efficient.
This can happen when all the columns needed for the query are part of the index,
either in the key columns or as included columns.

For example, let's assume we have the following table and index:

```sql
CREATE TABLE users (
  id INT(11) NOT NULL,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  email VARCHAR(50) NOT NULL,
  PRIMARY KEY (id),
  INDEX idx_name_email (first_name, last_name, email)
);
```

We can now run an index-only query by selecting only the columns available in
the index:

```sql
SELECT first_name, 
       last_name, 
       email
  FROM users
 WHERE first_name = 'John' AND 
       last_name = 'Doe';
```

This query can be executed using only the `idx_name_email` index, without
accessing the table data, leading to faster and more efficient query execution.

Index-only queries require all the necessary columns for the query to be present
in the index. This can lead to larger index sizes, taking up more disk space and
memory, which can negatively impact overall system performance.

## Filtering and Sorting With Joins

Let's say the number of tasks is 20,000 and the number of projects is 40 and the
number of joined records is 120,000 and the the number of result is 40.

```
                20,000                40
------> tasks ---------> projects ---------->
<------       <---------          <----------
   40          120,000
```

The number of filtered `tasks` should be small for better performance.

```sql
SELECT tasks.*
  FROM tasks
  JOIN projects USING(project_id)
 WHERE tasks.team_id = 4 AND 
       taks.status = 'open' AND 
       projects.status = 'open';
```

The number of filtered `invoices` should be small for better performacne.

```sql
SELECT *
  FROM invoices
  JOIN invoices_metadata USING(invoice_id)
 WHERE invoices.tenant_id = 4236
 ORDER BY invoices_metadata.due_date
 LIMIT 30
```

## Exceeding the Maximum Index Size

The size of index should be lower than data. Unless it will make following
error. `The index row size of 3480 bytes exceeds the maximum size of 2712 bytes for index contacts_fullname`

There are good strategy for index size optimization including **Prefix Index**, **Indexing Hash Values**, **Multiple Columns**.

**Prefix Index**

The main advantage of using a prefix index is that it significantly reduces the
size of the index, which in turn can speed up search operations. This can be
particularly beneficial when working with large volumes of text data, such as
with VARCHAR or TEXT fields.

```sql
CREATE INDEX articles_search ON articles (type, (substring(title, 1, 20)));
SELECT *
  FROM articles
 WHERE substring(title, 1, 20) = '...20 chars truncated title...' AND
       title = '...full title...'
```
However, there are some drawbacks and limitations to using prefix indexes:

* Prefix indexes can only be used with MyISAM and InnoDB storage engines. 
* When using a prefix index, MySQL may not be able to use the index for certain
  queries or operations like sorting or searching with the LIKE operator
  (depending on the query and storage engine). 
* Maintenance and updates to the index can be slower because it involves string
  comparisons.

```sql
CREATE INDEX articles_search ON articles (type, title(20));
SELECT * FROM articles WHERE title = '...full title...'
```

In this example, a prefix index named articles_search is created on the articles
table, including both the 'type' column and the 'title' column with a length of
20 characters. The storage engine will only index the first 20 characters of the
'title' field, which can be beneficial for optimizing the performance of search
queries on large VARCHAR or TEXT fields.  

This approach can improve the query execution speed by utilizing the prefix
index for efficient searching while still maintaining the accuracy of the
results based on the full title. However, keep in mind the considerations and
limitations of using a prefix index as mentioned in the previous response, such
as compatibility with specific storage engines and potential impact on index
maintenance.


**Indexing Hash Values**

An index based on hash values involves creating an index using a hash function,
such as SHA1, applied to the indexed column values. This can be useful for
speeding up search operations, especially for large text or binary columns.

```sql
CREATE INDEX articles_search ON articles (type, (sha1(title)));
SELECT *
  FROM articles
 WHERE sha1(title) = '...hash of title...' AND 
       title = '...full title...'
```

In the example you have provided, a hash index is created on the articles table
named `articles_search`, including the 'type' column and an SHA1 hash of the
'title' column. Applying the SHA1 hash function to the 'title' column values
helps to create a fixed-length hash value (40 characters), which can be used to
improve search performance and reduce the index size.

The subsequent SELECT statement uses both the SHA1 hash value and the full title
as search conditions in the query. The search operation first compares the SHA1
hash value through the index, which helps to quickly narrow down the potential
search results. Once the search is filtered based on the hash value, MySQL then
compares the full title to find the exact match for the query.

Using a hash index can improve search performance on large text columns, as the
hash values can be compared faster than the larger original text values.
However, there are some caveats and limitations to consider when using hash
indexes:

* In MySQL, hash index support is available only for the **MEMORY** storage engine,
  not for **MyISAM** or **InnoDB**, which limits its general use.
* Hash functions, such as **SHA1**, are not guaranteed to produce unique hash
  values for unique inputs (i.e., collisions can occur), which might result in
  false positives during the search. Including the full title comparison in the
  WHERE clause, as demonstrated in the example, helps ensure accurate results.
* Updates and modifications to the data may involve recalculating the hash
  value, as well as updating the index, affecting the performance of write
  operations.

```sql
CREATE INDEX articles_search ON articles (type, (SUBSTRING(sha1(title), 1, 20)));
SELECT *
  FROM articles
 WHERE SUBSTRING(sha1(title), 1, 20)) = '...shortened hash of title...' AND 
       title = '...full title...'
```

In this example, you are attempting to create an index on the articles table
named articles_search by incorporating the 'type' column and a truncated hash
value of the 'title' column. The truncation is achieved by applying the
SUBSTRING function to the result of the SHA1 hash function on the 'title',
reducing the hash value to a length of 20 characters. The idea behind this is to
create smaller indexed values which would potentially reduce the index size and
improve search performance.

However, creating an index in this manner is not directly supported by MySQL for
InnoDB or MyISAM storage engines in the way you have provided in your example.  

```sql
CREATE INDEX articles_search ON articles (type, (md5(title)::uuid));
SELECT *
  FROM articles
 WHERE md5(title)::uuid = md5('...full title...')::uuid AND 
       title = '...full title...'       
```

In this case, you are creating an index on the articles table named
articles_search, which includes the 'type' column and the MD5 hash of the
'title' column cast as a UUID (Universal Unique Identifier) data type. The
purpose is to create a fixed-length indexed value (128-bit) based on the hash,
which could potentially improve search performance. PostgreSQL supports the
`::uuid` casting operator, unlike MySQL.

However, even in PostgreSQL, there are important factors to consider when
relying on hashed column values for indexing:

* Hash collisions: Although rare, MD5 is not guaranteed to produce unique hash
  values for unique text inputs. Including the full title comparison in the
  WHERE clause helps to ensure the accuracy of the search results.
* Write performance: When any change is made to the 'title' column, the hash
  value and the index have to be updated, which may impact write performance.
* Query planning: In some cases, PostgreSQL’s query planner might not choose the
  index in query execution, and this could affect search performance.

**Multiple Columns**

Here are some key points to consider for using multiple columns for indexing
optimization:

- **Column order**: The order of columns in the index plays a crucial role, as it
  determines the search efficiency for different query patterns. In general, you
  should place columns with a higher level of selectivity, or columns with
  higher cardinality (i.e., a larger number of distinct values), earlier in the
  index.
- **Index efficiency**: For a multi-column index to be effective, the indexed
  WHERE, ORDER BY, or JOIN clauses. This helps in utilizing the index more
  columns should frequently appear together in query conditions, such as in
  efficiently, improving the performance of the query execution.
- **Covering index**: An index that includes all the columns needed for processing a
  query is called a covering index. When a query can be fulfilled entirely using
  the data stored in an index, the database engine doesn't have to access the
  underlying table, resulting in faster query execution.
- **Index size**: Multi-column indexes may have a smaller size compared to
  maintaining multiple single-column indexes, as the additional overhead for
  each index is reduced. A smaller index size also means less disk and memory
  usage, potentially improving performance.
- **Write performance**: Keep in mind that having multiple indexes or composite
  indexes on a table may affect the performance of INSERT, UPDATE, and DELETE
  operations since the indexes must be updated each time a relevant modification
  is made to the data. Finding the right balance between read performance and
  write performance is crucial.

```sql
CREATE INDEX articles_search ON articles (type, author, title);
```

In this example, a multi-column index named articles_search is created on the
articles table, comprising the 'type', 'author', and 'title' columns.

Overall, using multiple columns in an index can be beneficial for optimizing
size and improving query performance. However, it is important to consider the
specific needs and query patterns of your application to determine the optimal
index design. Be sure to analyze and monitor the index usage to continuously
fine-tune the indexing strategy.

## JSON Objects and Arrays

There 3 stretegy in JSO Object Indexing including **Virtual Columns**, 
**Functional Indexes**, **GIN Indexes**.

**Virtual Columns**

```sql
CREATE TABLE contacts (
  id bigint PRIMARY KEY AUTO_INCREMENT,
  attributes json NOT NULL,
  email varchar(255)  AS (attributes->>"$.email") VIRTUAL NOT NULL,
  INDEX contacts_email (email)
);
SELECT * 
  FROM contacts 
 WHERE attributes->>"$.email" ='admin@example.com';
```

Using virtual columns of JSON objects in MySQL provides the following advantages:

* Enhances the query performance of JSON data by creating indexes on specific
  JSON attributes using virtual columns.
* Enables better schema flexibility and design, by allowing a combination of
  structured and semi-structured data in one table.
* Reduces storage space requirements since virtual column values aren't
  physically stored and are only calculated when accessed.

However, it's important to consider the balance between performance gains and
potential computation overhead, as virtual columns are calculated at runtime. In
most cases, the overall benefits of using virtual columns to extract and index
JSON attributes outweigh the potential overhead.

**Functional Indexes**

```sql
-- PostgreSQL
CREATE INDEX contacts_email ON contacts ((attributes->>'email'));
SELECT * 
  FROM contacts 
 WHERE attributes->>'email' = 'admin@example.com';

-- MySQL
CREATE INDEX contacts_email ON contacts ((
  CAST(attributes->>"$.email" AS CHAR(255)) COLLATE utf8mb4_bin
));
SELECT * 
  FROM contacts  
 WHERE attributes->>"$.email" = 'admin@example.com';
```

Using a function index for JSON objects in SQL databases provides the following advantages:

* Enhances query performance by speeding up searches on specific JSON attributes
  through the use of indexes. 
* Enables better handling of JSON data by simplifying the query conditions for
  attribute filtering.
* Allows for indexing and searching on computed values, making complex queries
  more efficient. 
  
However, there are some potential trade-offs to consider when using function indexes:

* Function indexes may have an impact on write performance, as the index needs
  to be updated whenever there is a change in the indexed data.
* The computation of function-based index values can create some overhead in
  cases where the function is complex or computationally intensive.
* Syntax and support for function indexes may vary between different SQL
  database systems, so it is crucial to understand the specific capabilities and
  limitations of your chosen database.

**GIN Indexes**

```sql
CREATE INDEX contacts_attributes ON contacts USING gin (attributes);
```

A GIN (Generalized Inverted Index) index is a type of index available in
**PostgreSQL** that is specifically designed to index complex data types, such
as arrays, full-text search tsvector, and JSON/JSONB objects. GIN indexes
provide high-performance search capabilities on such complex data fields and are
especially useful for searching JSON data based on key-value pairs, keys, or
values.

Using GIN indexes for JSON objects in PostgreSQL provides the following advantages:

* Enhances query performance for complex JSON queries, such as containment and
  existence queries, by utilizing a high-performance search index.
* Provides an efficient way to index key-value pairs, keys, or values in JSON
  objects. Suits well for read-heavy workloads and infrequent writes.
* Supports various query types and operators, making it suitable for different
  use cases when working with JSON data.

However, it's important to consider some potential trade-offs when using GIN indexes:

* **GIN** indexes may have an impact on write performance since the index needs to
  be updated whenever there's a change in the indexed data.
* **GIN** indexes consume more storage space compared to other index types, such as
  **B-tree** or **GiST** indexes.
* Building a **GIN** index can be time-consuming, especially for large data sets.

```sql
-- Containment query:
SELECT * FROM contacts WHERE attributes @> '{"email": "admin@example.com"}';
-- This query returns all rows where the attributes JSON object contains an 
-- 'email' key with the value "admin@example.com". The @> operator checks 
-- if the JSON on the left contains the JSON on the right.

-- Existence query:
SELECT * FROM contacts WHERE attributes ? 'email';
-- This query returns all rows with the 'email' key present 
-- in the attributes JSON object. The ? operator checks if the key 
-- on the right exists in the JSON object on the left.

-- Key existence (OR) query:
SELECT * FROM contacts WHERE attributes ?| array['email', 'phone'];
-- This query returns all rows where either 'email' key or 'phone' key 
-- is present in the attributes JSON object. The ?| operator checks 
-- if any key in the array on the right exists in the JSON object on the left.

-- Key existence (AND) query:
SELECT * FROM contacts WHERE attributes ?& array['email', 'phone'];
-- This query returns all rows where both 'email' key and 'phone' 
-- key are present in the attributes JSON object. The ?& operator checks 
-- if all keys in the array on the right exist in the JSON object on the left.

-- JSON path query:
SELECT * FROM contacts WHERE attributes @? '$.tags[*] ? (@.type == "note" && @.severity > 3)';
-- This query uses the JSON path language, specified by the '@?' operator, 
-- to filter rows based on more complex criteria. In this case, 
-- it returns all rows where the attributes JSON data has elements 
-- in the 'tags' array with a 'type' key equal to "note" and 
-- a 'severity' key with a value greater than 3.
```

**JSON Arrays**

```sql
-- PostgreSQL
CREATE INDEX products_categories ON products USING gin(categories);
SELECT * 
  FROM products 
 WHERE categories @> '["printed book", "ebook"]' AND 
       NOT categories @> '["audiobook"]';

-- MySQL
CREATE INDEX products_categories on products ((CAST(categories AS UNSIGNED ARRAY)));
SELECT * 
  FROM contacts 
 WHERE JSON_CONTAINS(attributes, CAST('[17, 23]' AS JSON)) AND 
       NOT JSON_CONTAINS(attributes, CAST('[11]' AS JSON))
```

## Unique Indexes and Null

When dealing with unique indexes, NULL values are considered distinct from each
other. This means that a unique index allows multiple rows to have NULL values
in the indexed column(s) without violating the uniqueness constraint.

```sql
-- PostgreSQL
-- Creating the 'users' table
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  alternate_email VARCHAR(255)
);

-- Creating a unique index on the 'email' column
CREATE UNIQUE INDEX users_email ON users (email);

-- Creating a unique index on the 'alternate_email' column
CREATE UNIQUE INDEX users_alternate_email ON users (alternate_email);

-- Inserting data (valid)
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
INSERT INTO users (email, alternate_email) VALUES ('user2@example.com', NULL);

-- Inserting a duplicate value (violates unique constraint)
-- This will fail
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
```

```sql
-- MySQL
-- Creating the 'users' table
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  alternate_email VARCHAR(255),
  UNIQUE INDEX users_email (email),
  UNIQUE INDEX users_alternate_email (alternate_email)
);

-- Inserting data (valid)
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
INSERT INTO users (email, alternate_email) VALUES ('user2@example.com', NULL);

-- Inserting a duplicate value (violates unique constraint)
-- This will fail
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
```

This will fix that problem which does not suits for not null unique index.

```sql
-- Creating table
CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  customer_id INT,
  shipment_id INT
);

-- Creating a partial unique index for non-null customer_id and shipment_id
CREATE UNIQUE INDEX uniqueness_idx_customer_shipment_not_null ON orders (
  customer_id, shipment_id
) WHERE customer_id IS NOT NULL AND shipment_id IS NOT NULL;

-- Adding a check constraint to prevent both columns from being NULL at the same time
ALTER TABLE orders ADD CONSTRAINT chk_customer_shipment_not_both_null CHECK (
  customer_id IS NOT NULL OR shipment_id IS NOT NULL
);
```

This will fix not null discint problem.

```sql
CREATE UNIQUE INDEX uniqueness_idx ON orders (
  customer_id,
  (CASE WHEN shipment_id IS NULL THEN -1 ELSE shipment_id END)
);
```

However, there are several considerations when using this approach:

- The solution assumes that the replaced value (-1, in this case) will never be
  used as a valid value for the `shipment_id` column, as it might lead to unexpected
  behavior when inserting or updating data. You need to ensure that this value is
  not used accidentally as an actual value.
- Replacing the `NULL` values with distinct values like this can be seen as a bit
  "hacky" and might introduce some confusion when maintaining or debugging the
  database schema in the future.

## Location-Based Searching With Bounding-Boxes

```sql
CREATE TABLE businesses (
  id bigint PRIMARY KEY NOT NULL,
  type varchar(255) NOT NULL,
  name varchar(255) NOT NULL,
  latitude float NOT NULL,
  longitude float NOT NULL
);
CREATE INDEX search_idx ON businesses (longitude, latitude);

SELECT *
  FROM businesses
 WHERE type = 'restaurant' longitude BETWEEN -73.9752 AND -74.0083 AND 
       latitude BETWEEN 40.7216 AND 40.7422
```

Location-based searching with bounding boxes is a technique used to search and
filter results based on their geographical locations. This can be useful for
applications like restaurant finders, real estate websites, and
geolocation-based search engines. Bounding boxes are defined as rectangular
areas within which a search is performed, and they are typically specified by
their top-left and bottom-right coordinates (latitude and longitude).

In the example provided, we have a table called 'businesses' with columns 'id',
'type', 'name', 'latitude', and 'longitude'. We create an index named
'search_idx' on the 'longitude' and 'latitude' columns to improve the
performance of location-based queries.

The SQL query in the example searches for businesses with the type 'restaurant'
located within specified longitude and latitude boundaries. The longitude
boundaries are defined as -73.9752 (left) and -74.0083 (right), while the
latitude boundaries are set to 40.7216 (bottom) and 40.7422 (top). This query
will return all restaurants within the specified bounding box.

To execute efficient location-based searches in SQL databases like MySQL and
PostgreSQL, it is essential to make use of spatial indexing techniques such as
`R-trees` or `GiST` (Generalized Search Tree) for faster retrieval of records based
on their geographical location. Using bounding boxes is a simple but effective
method for narrowing down results on location-based searches, allowing for
efficient and fast retrieval of relevant information.

location-based searching involves finding and filtering records based on their
geographical locations, which is particularly useful for applications like
restaurant finders, real estate websites, and geolocation-based search engines.
Several methods and data structures can be used to improve the efficiency of
location-based searches, such as:

- Bounding Boxes: A simple technique where a rectangular area is defined by its
  top-left and bottom-right coordinates, and searching is performed within that
  area.
- Quadtrees: A tree data structure used for partitioning a two-dimensional space
  by recursively subdividing it into four equal quadrants or regions, enabling
  efficient storage and retrieval of spatial data.
- R-tree: A spatial indexing method that uses bounding rectangles with variable
  sizes and overlaps to partition the space, improving search performance for
  datasets with irregular spatial distribution.
- k-d tree: A tree data structure for partitioning k-dimensional space,
  particularly efficient for nearest neighbor and point-radius searches.
- Geohash: A geocoding system using short alphanumeric strings to represent
  rectangular areas on Earth, based on Z-order curve space-filling techniques,
  useful for searching in distributed systems or big data platforms.
- Spatial Partitioning: Dividing the dataset into pre-defined partitions or
  zones according to geographical boundaries for more efficient search
  performance.
- Haversine Formula: A formula for calculating the great-circle distance between
  two points on Earth's surface, used for filtering search results based on the
  distance between the user's location and the records in the dataset.
