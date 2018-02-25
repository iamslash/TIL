# Abstract

stream editor에 대해 적는다.

# References

* [Sed 가이드](https://mug896.gitbooks.io/sed-script/content/)
  * 친절한 한글
* [Sed - An Introduction and Tutorial by Bruce Barnett](http://www.grymoire.com/Unix/Sed.html#uh-0)
* [sed @ gnu](https://www.gnu.org/software/sed/manual/sed.html)
* [부록 B. Sed 와 Awk 에 대한 간단한 입문서](https://wiki.kldp.org/HOWTO/html/Adv-Bash-Scr-HOWTO/sedawk.html)

# Intro

sed command line은 보통 다음과 같은 형식을 같는다.

```bash
sed SCRIPT INPUTFILE...
```

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
