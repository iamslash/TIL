# intro

- NoSql db중 document type의 db이다.

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

## transaction (two phase commit) 

## aggregation

### aggregation pipeline

### map-reduce function

### single purpose aggregation methods

## replication

## sharding



## Memory Mapping

# Reference

- [mongodb manual](https://docs.mongodb.com/manual/)
  - 최고의 문서는 메뉴얼!!! 목차 내용은 필수