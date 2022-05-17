- [Abstract](#abstract)
- [Materials](#materials)
- [MySQL Lock](#mysql-lock)
- [Shared and Exclusive Locks](#shared-and-exclusive-locks)
- [Intention Locks](#intention-locks)
- [Record Locks](#record-locks)
- [Gap Locks](#gap-locks)
- [Next-Key Locks](#next-key-locks)
- [Insert Intention Locks](#insert-intention-locks)
- [AUTO-INC Locks](#auto-inc-locks)
- [Predicate Locks for Spatial Indexes](#predicate-locks-for-spatial-indexes)
- [Inno-db Deadlock](#inno-db-deadlock)

----

# Abstract

MySQL InnoDB 의 Lock 에 대해 정리한다.  

# Materials

> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.1 InnoDB Locking @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-locking.html)
> * [InnoDB locking](https://github.com/octachrome/innodb-locks)

# MySQL Lock

mysql 의 innodb 는 다음과 같은 종류의 lock 을 갖는다.

* Shared and Exclusive Locks
* Intention Locks
* Record Locks
* Gap Locks
* Next-Key Locks
* Insert Intention Locks
* AUTO-INC Locks
* Predicate Locks for Spatial Indexes

# Shared and Exclusive Locks

InnoDB implements standard row-level locking where there are two types of locks,
shared `(S)` locks and exclusive `(X)` locks.

* A shared (S) lock permits the transaction that holds the lock to read a row.
  * `SELECT ... LOCK IN SHARE MODE`
  * A request by T2 for an `S` lock can be granted immediately. As a result, both
    T1 and T2 hold an `S` lock on r.
* An exclusive (X) lock permits the transaction that holds the lock to update or
delete a row.
  * A request by T2 for an `X` lock cannot be granted immediately.
  * `SELECT ... FOR UPDATE`

# Intention Locks

The main purpose of intention locks is to show that someone is locking a row, or
going to lock a row in the table.

* An intention shared lock `(IS)` indicates that a transaction intends to set a
  shared lock on individual rows in a table.
  * `SELECT ... LOCK IN SHARE MODE`
  * Before a transaction can acquire a shared lock on a row in a table, it must
    first acquire an `IS` lock or stronger on the table.
* An intention exclusive lock `(IX)` indicates that a transaction intends to set
  an exclusive lock on individual rows in a table.
  * `SELECT ... FOR UPDATE`
  * Before a transaction can acquire an exclusive lock on a row in a table, it
    must first acquire an `IX` lock on the table.

# Record Locks

A record lock is a lock on an index record. For example, `SELECT c1 FROM t WHERE c1 = 10 FOR UPDATE;` prevents any other transaction from inserting, updating, or
deleting rows where the value of `t.c1` is `10`.

```sql
> SHOW ENGINE INNODB STATUS;
RECORD LOCKS space id 58 page no 3 n bits 72 index `PRIMARY` of table `test`.`t`
trx id 10078 lock_mode X locks rec but not gap
Record lock, heap no 2 PHYSICAL RECORD: n_fields 3; compact format; info bits 0
 0: len 4; hex 8000000a; asc     ;;
 1: len 6; hex 00000000274f; asc     'O;;
 2: len 7; hex b60000019d0110; asc        ;;
```

# Gap Locks

A gap lock is a lock on a gap between index records, or a lock on the gap before
the first or after the last index record. For example, SELECT c1 FROM t WHERE c1
BETWEEN 10 and 20 FOR UPDATE; prevents other transactions from inserting a value
of 15 into column t.c1, whether or not there was already any such value in the
column, because the gaps between all existing values in the range are locked.

# Next-Key Locks

A next-key lock is a combination of **a record lock** on the index record and** a gap
lock** on the gap before the index record.

# Insert Intention Locks

An insert intention lock is a type of gap lock set by INSERT operations prior to
row insertion. This lock signals the intent to insert in such a way that
multiple transactions inserting into the same index gap need not wait for each
other if they are not inserting at the same position within the gap. 

# AUTO-INC Locks

An AUTO-INC lock is a special table-level lock taken by transactions inserting
into tables with AUTO_INCREMENT columns.

# Predicate Locks for Spatial Indexes

InnoDB supports SPATIAL indexing of columns containing spatial columns???

# Inno-db Deadlock

> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.5.1 An InnoDB Deadlock Example @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-deadlock-example.html)

mysql 의 innodb 는 Deadlock 을 detect 할 수 있다. 만약 mysql 이 Deadlock 을 detect 하면
어느 한 transaction 의 lock wait 을 중지하여 Deadlock 을 해결한다.
