# Install with docker

```console
$ docker run -p 5432:5432 -e POSTGRES_PASSWORD=1 -e POSTGRES_USER=iamslash -e POSTGRES_DB=basicdb --name my-postgres -d postgres

$ docker exec -i -t my-postgres

$ su - postgres

$ psql basicdb
\list
\dt
SELECT * FROM account;
```

# Integration with PostgreSQL

* [expostgres @ spring-examples](https://github.com/iamslash/spring-examples/tree/master/expostgres)


