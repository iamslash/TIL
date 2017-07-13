# abstract

- mysqld에 대해 적어보자.

# tip

## how to reset password

```
pkill mysqld
sudo /usr/local/mysql/bin/mysqld_safe --skip-grant-tables
mysqld
mysql -u root
> GRANT ALL PRIVILEGES ON *.* to 'root'@'localhost' WITH GRANT OPTION
> FLUSH PRIVILEGES;

```

## how to run multiple mysqld instances

```
mysqld_multi
```

## XA

