- [Abstract](#abstract)
- [Basic](#basic)
- [Addresses](#addresses)
- [AND, OR](#and-or)
- [Commands](#commands)
- [Branch](#branch)
- [Multiple lines](#multiple-lines)
- [Execute](#execute)
- [Command line options](#command-line-options)
- [Debugging](#debugging)
- [References](#references)
- [Advanced](#advanced)
  - [Replace in files](#replace-in-files)
  - [두개의 패턴 사이의 내용을 지우자.](#두개의-패턴-사이의-내용을-지우자)

-------------------------------------------------------------------------------

# Abstract

unix 가 만들어 졌을때 적은 메모리에도 동작할 수 있는 라인 에디터가 필요해서
`/bin/ed` 가 탄생했다. ed 는 이후 `sed`, `ex`, `vi`, `grep` 등의 기반이 된다.
`sed` 는 `ed` 의 stream 버전이다. 명령어 사용법이 같다.

`ex` 는 `ed` 의 확장버전이다. `vi` 에서 `:` 를 이용한 command mode 로 사용된다.
`ex` 의 visual mode version 이 `vi` 이다.

# Basic

sed command line 은 보통 다음과 같은 형식을 같는다.

```bash
sed SCRIPT INPUTFILE...
```

sed 는 **pattern space** 와 **hold space** 라는 두가지 버퍼가 존재한다.  하나의
줄을 읽으면 **pattern space** 에 저장하고 필요할 때마다 **hold space** 에 저장해
둔다. **hold space** 에 저장해두면 명령어에 적용되지 않는다. 아무런 옵션이 없다면
**pattern space** 에 저장된 줄을 출력한다.

```bash
$ seq 111 111 555 | sed ''
111
222
333
444
555
# 2번째 줄을 삭제한다.
$ seq 111 111 555 | sed '2d'
111
333
444
555
# 3번째 줄만 출력한다. 나머지는 삭제한다.
$ seq 111 111 555 | sed '1d; 2d; 4d; 5d'
333
# -n 옵션은 출력을 허용하지 않는다. p 명령으로 출력한다.
$ seq 111 111 555 | sed -n '3p'
333
```

`{}` 를 이용하여 command group 을 만들자. `;` 를 이용하여 command list 를
만들자. `{}` 뒤에 `;` 를 붙여야 한다.

```bash
# 해당 address 에 여러 개의 명령을 사용하려면 { } 를 사용합니다.
$ seq 5 | sed -n '2p; 4{p;p}; $p'
2
4
4
5

# 명령들이 각 라인별로 위치할 경우는 `;` 를 붙이지 않아도 된다.
$ sed -f - <<\EOF datafile
    /"crosshair"/ {
        :X                          # branch 명령을 위한 label
        N
        /}/ {                       # { } 명령 그룹의 사용
            /sprites\/crosshairs/d
            b                       # 명령 사이클의 END 로 branch
        }
        bX                          # label :X 로 branch
    }
EOF
```

# Addresses

# AND, OR

# Commands

# Branch

# Multiple lines

# Execute

# Command line options

# Debugging

# References

* [Sed 가이드](https://mug896.gitbooks.io/sed-script/content/)
  * 친절한 한글
* [Sed - An Introduction and Tutorial by Bruce Barnett](http://www.grymoire.com/Unix/Sed.html#uh-0)
* [sed @ gnu](https://www.gnu.org/software/sed/manual/sed.html)
* [부록 B. Sed 와 Awk 에 대한 간단한 입문서](https://wiki.kldp.org/HOWTO/html/Adv-Bash-Scr-HOWTO/sedawk.html)

# Advanced

## Replace in files

```bash
$ find . -type f -name 'config.xml' | xargs sed -i 's/<disabled>false<\/disabled>/<disabled>true<\/disabled>/g'
$ find . -type f -name 'config.xml' | xargs grep '<disabled>false</disabled>'
```

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
  * [참고](https://nixtricks.wordpress.com/2013/01/09/sed-delete-the-lines-lying-in-between-two-patterns/)

```bash
sed '/PATTERN-1/,$d' a.txt
```


