# Abstract

java 에서 사용할 만한 DB migration tool

# Materials

* [Flyway 개념 & 사용법](https://bkim.tistory.com/2)
* [Flyway getting started](https://flywaydb.org/getstarted/firststeps/commandline.html)
* [SpringBoot and Database migrations with FlyWay](https://www.youtube.com/watch?v=_7BuLOCRJc4)
* [Flyway 공부](http://chanwookpark.github.io/dbmigration/flyway/2016/08/23/flyway/)

# Install

[Flyway](https://flywaydb.org/download/) 에서 download 받고 압축을 푼다.

# Basic

## SQL rule

* [SQL-based migrations](https://flywaydb.org/documentation/migrations#sql-based-migrations)

SQL 을 작성할 때는 반드시 다음 규칙을 지켜야 한다.

* 파일명은 V 와 number 로 시작한다. 
* 그리고 2 개의 under score 가 있어야 한다.
* 의미있는 문장과 마지막은 sql 로 끝난다.

## Commands

* [migrate](https://flywaydb.org/documentation/command/migrate)
* [clean](https://flywaydb.org/documentation/command/clean)
* [info](https://flywaydb.org/documentation/command/info)
* [validate](https://flywaydb.org/documentation/command/validate)
* [undo](https://flywaydb.org/documentation/command/undo)
  * If target is specified, Flyway will attempt to undo versioned migrations in the order they were applied until it hits one with a version below the target. If group is active, Flyway will attempt to undo all these migrations within a single transaction.
* [baseline](https://flywaydb.org/documentation/command/baseline)
  * Baseline is for introducing Flyway to existing databases by baselining them at a specific version. 
* [repair](https://flywaydb.org/documentation/command/repair)
  * Repair is your tool to fix issues with the schema history table. It has two main uses:

## Tutorial

```console

$ docker run -p3306:3306 --rm --name my-mysql -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=foo -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=1 -d mysql

$ cd ~/tmp/flyway-6.4.1

$ vim conf/flyway.conf
flyway.url=jdbc:mysql://localhost:3306/foo
flyway.user=iamslash
flyway.password=1

$ vim sql/V1__Create_person_table.sql
CREATE TABLE 

create table PERSON ( 
    ID int not null, 
    NAME varchar(100) not null 
);

insert into PERSON (ID, NAME) values (1, 'Axel'); 
insert into PERSON (ID, NAME) values (2, 'Mr. Foo'); 
insert into PERSON (ID, NAME) values (3, 'Ms. Bar');

$ flyway migrate

$ flyway info
+-----------+---------+---------------------+------+---------------------+---------+
| Category  | Version | Description         | Type | Installed On        | State   |
+-----------+---------+---------------------+------+---------------------+---------+
| Versioned | 1       | Create person table | SQL  | 2020-05-10 07:39:17 | Failed  |
| Versioned | 2       | Add people          | SQL  |                     | Pending |
+-----------+---------+---------------------+------+---------------------+---------+

$ vim sql/V2__Add_people.sql
insert into PERSON (ID, NAME) values (1, 'Axel');
insert into PERSON (ID, NAME) values (2, 'Mr. Foo');
insert into PERSON (ID, NAME) values (3, 'Ms. Bar');

$ vim sql/V1__Create_person_table.sql
create table PERSON ( 
    ID int not null, 
    NAME varchar(100) not null 
);

$ flyway clean
$ flyway migrate

$ vim sql/V3__Add_people.sql
insert into PERSON (ID, NAME) values (5, 'Slash');
insert into PERSON (ID, NAME) values (6, 'PB');

$ flyway validate

$ flyway info
+-----------+---------+---------------------+------+---------------------+---------+
| Category  | Version | Description         | Type | Installed On        | State   |
+-----------+---------+---------------------+------+---------------------+---------+
| Versioned | 1       | Create person table | SQL  | 2020-05-10 07:42:25 | Success |
| Versioned | 2       | Add people          | SQL  | 2020-05-10 07:42:25 | Success |
| Versioned | 3       | Add people          | SQL  |                     | Pending |
+-----------+---------+---------------------+------+---------------------+---------+

$ flyway migrate

$ flyway clean
```
