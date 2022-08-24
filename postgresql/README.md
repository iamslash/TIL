- [Materials](#materials)
- [Install](#install)
- [Basic](#basic)
  - [Integration with PostgreSQL](#integration-with-postgresql)
  - [vaccuum](#vaccuum)

----

# Materials

* [crunchdata](https://www.crunchydata.com/developers/tutorials)
  * browser 에서 PostgreSQL 을 띄우고 공부할 수 있다.
* [The Art of PostgreSQL](https://theartofpostgresql.com/)
  * mater piece

# Install

```console
$ docker run -p 5432:5432 -e POSTGRES_PASSWORD=1 -e POSTGRES_USER=iamslash -e POSTGRES_DB=basicdb --name my-postgres -d postgres

$ docker exec -i -t my-postgres

$ su - postgres

$ psql basicdb
\list
\dt
SELECT * FROM account;
```

# Basic

## Integration with PostgreSQL

* [expostgres @ spring-examples](https://github.com/iamslash/spring-examples/tree/master/expostgres)

## vaccuum

* [PostgreSQL: 베큠(VACUUM)을 실행해야되는 이유 그리고 성능 향상](https://blog.gaerae.com/2015/09/postgresql-vacuum-fsm.html)

----

FSM (Free Space Map) 에 쌓여진 데이터를 지우는 것. 디스크 조각모으기와 비슷하다.
