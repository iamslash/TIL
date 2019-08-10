# Abstract

- 3.4를 기준으로 정리한다. 메뉴얼이 정말 잘 되어 있다. 서비스에 이용한다면 꼭 필독하자.
- NoSql db중 document type의 db이다.
- single-document transaction은 지원하지만 multi-document
  transaction은 지원하지 않는다. two phase commit을 이용해
  multi-document transaction을 구현할 수 있다.
- muli-granularity locking을 사용하여 global, database, collection level의 lock을 한다.
  - storage engine 별로 collection 미만 level의 lock을 한다.
  - WiredTiger는 document level lock을 한다.
  - granularity locking은 [이곳](http://www.mysqlkorea.com/sub.html?mcode=manual&scode=01&m_no=21879&cat1=14&cat2=422&cat3=444&lang=k) 을 참고해서 이해하자.

# install

- brew install mongodb


# terms and definition

| mongodb    |      RDBMS    | 
|:----------|-------------|
| _ID Field  |  Primary Key |
| BSON Field | Column |
| Collection | Table |
| BSON Document | Row |
| Embedded & Linking | Relation Ship |


# usage

## Query

```
> p = { eno : 1101, fname : "Adam", Iname : "Kroll", job : "Manager", salaray : 100000, dept_name : "SALES" }
> db.emp.save(p)

> db.createCollection("emp", { capped : false, size : 8192});
> show collections
> db.emp.validate()
> db.emp.renameCollection("employees")
> db.employees.drop()

> db.emp.insert({ eno : 1101, fname : "JIMMY" });
> db.emp.insert({ eno : 1102, fname : "ADAM", iname : "KROLL" });
> db.emp.insert({ eno : 1103, fname : "SMITH", job : "CLERK" });
> db.emp.update({ eno : 1101 }, { $set: {fname : "JOO"} } );
> db.emp.update({ eno : 1102 }, { $set: {job : "CHIEF" } } );
> db.emp.update({ eno : 1103 }, { $set: {iname : "STANDFORD" } } );
> db.emp.find().sort({ eno : -1 });
> db.emp.remove({eno: 1101});
```

## SQL vs mongo query

| SQL    |      mongo query    | 
|:----------|:-------------|
| CREATE TABLE emp (empno Number, ename Number)  | db.createCollection("emp") |
| INSERT INTO emp VALUES(3, 5) | db.emp.insert({empno: 3, ename: 5}) |
| SELECT * FROM emp | db.emp.find() |
| SELECT empno, ename FROM emp | db.emp.find({}, {empno: 1, ename: 1}) |
| SELECT * FROM emp WHERE empno = 3 | db.emp.find({empno: 3}) |
| SELECT empno, ename FROM emp WHERE empno = 3 | db.emp.find({empno: 3}, {empno: 1, ename: 1}) |
| SELECT * FROM emp WHERE empno = 3 ORDER BY ename | db.emp.find({empno: 3}).sort({ename: 1}) |
| SELECT * FROM emp WHERE empno > 3 | db.emp.find({empno: {$gt: 3}}) |
| SELECT * FROM emp WHERE empno != 3 | db.emp.find({empno: {$ne: 3}}) |
| SELECT * FROM emp WHERE ename LIKE "%Joe%" | db.emp.find({ename: /Joe/}) |
| SELECT * FROM emp WHERE ename like "JOE%" | db.emp.find({ename: /^Joe/}) |
| SELECT * FROM emp WHERE empno > 1 AND empno <= 4 | db.emp.find({empno: {$gt: 1, $lte: 3}}) |
| SELECT * FROM emp ORDER BY ename DESC | db.emp.find().sort({ename: -1}) |
| SELECT * FROM emp WHERE empno = 1 AND ename = 'Joe' | db.emp.find({empno: 1, ename: 'Joe'}) |
| SELECT * FROM emp WHERE empno = 1 OR empno = 3 | db.emp.find({$or: [{empno: 1}, {empno: 3}]}) |
| SELECT * FROM emp WHERE rownum = 1 | db.emp.findOne() |
| SELECT empno FROM emp o, dept d WHERE d.deptno = o.deptno AND d.deptno = 10 | o = db.emp.findOne({empno: 1}); name = db.dept.findOne({deptno: o.deptno}); |
| SELECT DISTINCT ename FROM emp | db.emp.distinct('ename') |
| SELECT COUNT(*) FROM emp | db.emp.count() |
| SELECT COUNT(*) FROM emp WHERE deptno > 10 | db.emp.find({deptno: {$gt: 10}}).count() |
| SELECT COUNT(sal) FROM emp | db.emp.find({sal: {$exists: true}}).count() |
| CREATE INDEX i_emp_ename ON emp(ename) | db.emp.ensureIndex({ename: 1}) |
| CREATE INDEX i_emp_no ON emp(deptno ASC, ename DESC) | db.emp.ensureIndex({depno: 1, ename: -1}) |
| UPDATE emp SET ename = 'test' WHERE empno = 1 | db.emp.update({empno: 1}, {$set: {ename: 'test'}}) |
| DELETE FROM emp WHERE deptno = 10 | db.emp.remove({deptno: 10}) |

## aggregation

- 다양한 document들을 그룹화 하고 그룹화 한것을 하나의 결과로 리턴하는 기능
- aggregation pipeline, map-reduce function, single purpose
  aggregation methods 와 같이 총 3가지를 지원한다.

## Text Search

- text index와 $text operator를 이용하여 효율적으로 문자열 검색을 할 수 있다.

## Data Models

- mongodb는 기본적으로 schemaless하다. 하지만 validator를 이용해서
  document의 schema를 검사 할 수 있다.
- mongodb는 기본적으로 embedded data model과 normalized data model등
  두가지 data model을 갖는다. embedded ddata model은 특정 document에서
  다른 document를 내부에서 소유하는 것이고 normlized data model은 다른
  document의 _id만 소유하는 것이다.

## indexes

- text를 위해 text index가 있다.
- geospatial query를 위해 2dsphere index가 있다.


## Storage

- WiredTiger, MMAPv1, In-Memory Storage Engine이 있다.
- journal은 monbodb의 장애상황이 복구할 정보가 담겨 있는 로그이다.
- 하나의 docuemtn는 16MB를 초과 할 수 없다. 그런 경우에는 GridFS를
  사용해야 한다.

# Reference

- [mongodb manual](https://docs.mongodb.com/manual/)
  - 최고의 문서는 메뉴얼!!! 목차 내용은 필수

