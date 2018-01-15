# Abstract

# References

* [Sed - An Introduction and Tutorial by Bruce Barnett](http://www.grymoire.com/Unix/Sed.html#uh-0)

# Tips

## 두개의 패턴 사이의 내용을 지우자.

* 패턴을 포함하지 않은 내용을 지우고 출력하자.

```bash
sed '/PATTERN-1/,/PATTERN-2/{//!d}' a.txt
```

* 패턴을 포함하지 않은 내용을 지우고 파일을 수정하자.

```bash
sed -i '/PATTERN-1/,/PATTERN-2/{//!d}' a.txt
```

* 패턴을 포함한 내용을 지우고 출력하자.

```bash
sed '/PATTERN-1/,/PATTERN-2/d' a.txt
```

* 패턴 다음 내용을 모두 지우자.

```bash
sed '/PATTERN-1/,$d' a.txt
```

* [참고](https://nixtricks.wordpress.com/2013/01/09/sed-delete-the-lines-lying-in-between-two-patterns/)
