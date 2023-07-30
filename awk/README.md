- [Abstract](#abstract)
- [References](#references)
- [Basic](#basic)
  - [Basic Usage](#basic-usage)
  - [Data Types](#data-types)
  - [Variables](#variables)
  - [Regexp](#regexp)
  - [Functions](#functions)
  - [BEGIN, END](#begin-end)
  - [Builtin Variables](#builtin-variables)
    - [RECORD](#record)
    - [FIELD](#field)
    - [FORMAT](#format)
    - [MATCHING](#matching)
    - [ARGUMENT](#argument)
    - [ARRAY](#array)
    - [ETC](#etc)
  - [Record seperation](#record-seperation)
  - [Field seperation](#field-seperation)
  - [`$0=$0, $1=$1`](#00-11)
  - [Redirection](#redirection)
  - [getline](#getline)
  - [Arrays](#arrays)
  - [Control statements](#control-statements)
  - [Operators](#operators)
  - [Builtin Functions](#builtin-functions)
    - [Numeric Functions](#numeric-functions)
    - [String Functions](#string-functions)
    - [I/O Functions](#io-functions)
    - [Time Functions](#time-functions)
    - [Bitweise Functions](#bitweise-functions)
    - [Type Functions](#type-functions)
    - [I18N Functions](#i18n-functions)
  - [TCP/IP](#tcpip)
  - [Debugging](#debugging)
- [Advanced](#advanced)
  - [필드 출력하기](#필드-출력하기)


-----

# Abstract

c문법을 연상시키는 텍스트 처리 언어이다. 로그 파일등을 읽어서 원하는 정보를
선택하여 출력할 때 사용한다.

# References

* [awk 스크립트 가이드](https://mug896.gitbooks.io/awk-script/content/)
  * 친절한 한글
* [awk manual | gnu](https://www.gnu.org/software/gawk/manual/gawk.html#Getting-Started)
* [부록 B. Sed 와 Awk 에 대한 간단한 입문서](https://wiki.kldp.org/HOWTO/html/Adv-Bash-Scr-HOWTO/sedawk.html)

# Basic

## Basic Usage

awk command line은 보통 다음과 같은 형식을 갖는다.

```bash
awk 'program' input-file1 input-file2 ...
```

다음은 `df` 출력의 네 번째 항목을 `awk`와 `sed`를 사용하여 출력하는
예이다. `awk`가 훨씬 간단하다. 

```
$ df -h
$ df -h | sed -rn '2s/(\S+\s+){3}(\S+).*/\2/p'
# NR==2 는 2번째 record, $4는 4번째 column을 의미한다.
$ df -h | awk 'NR==2{ print $4 }'
```

`a.txt`를 읽어서 6번째 컬럼에 10을 더해서 출력한다.

```bash
$ awk '{ $6 += 10 }1' a.txt
```

`sed`는 기본적으로 출력하지만 `awk`는 그렇지 않다.

```bash
$ seq 3 | sed '' | column
1 2 3
# -n 옵션을 이용하여 출력모드를 조절할 수 있다.
$ seq 3 | sed -n ''
$
$ seq 3 | awk ''
```

`awk`는 표현식이 참일 경우 `$0`값이 출력된다.

```bash
$ seq 5 | awk 0
$
$ seq 5 | awk 1 | column
1 2 3 4 5
# 표현식이 참이면 무엇이든 사용할 수 있다.
$ seq 5 | awk 'NR==3'
3
$ seq 5 | awk '$1 > 2'
3
4
5
$ seq 5 | awk '/2/'
2
# '1' 은 사실상 '{print}' 와 같다.
$ seq 3 | awk '{ gsub(/.*/,"__&__") } 1'
__1__
__2__
__3__
```

다음은 표현식이 거짓인 경우이다.

```bash
# foo는 unset이다.
$ awk 'BEGIN{ if (foo) { print 111 } }'
# foo는 null이다.
$ awk 'BEGIN{ foo = ''; if (foo) { print 111 } }'
$ seq 3 | awk '""'
$ seq 3 | awk 'a = ""'
# 0은 거짓이다.
$ awk 'BEGIN{ if (0) {print 111 } }'
$ awk 'BEGIN{ v = 2; if (v - 2) { print 111 } }'
```

다음은 표현식이 참인 경우이다.

```bash
$ awk 'BEGIN{ if (2) { print 111 } }'
111
$ awk 'BEGIN{ if (-1) { print 111 } }'
111
$ awk 'BEGIN{ if (0.00001} { print 111 } }'
111
$ awk 'BEGIN{ if ("abc") { print 111 } }'
111
$ awk "BEGIN{ if ("0") { print 111 } }'
111
$ awk "BEGIN( if (" ") { print 111 } }'
111
$ seq 2 | awk '" "'
1
2
$ seq 2 | awk 'a = " "'
1
2
$ awk 'BEGINFILE{ if (ERRNO) { ... } }'
$ echo foobar | awk '{ if (/oba/) print 111 }'
111
```

공백라인은 NF가 0이기 때문에 출력되지 않는다.

```
$ awk 'NF' a.txt
```

shell command line은 참일 경우 0을 반환하기 때문에 다음과 같이 0과
비교한다.

```bash
$ awk 'BEGIN { if (system("test -f foo -a -x foo") == 0) { ... } }'
```

logical not은 `!=`, `!~`, `!`을 사용한다.

```bash
$ seq 3 | awk 'NR == 2'
2
$ seq 3 | awk 'NR != 2'
1
2
$ seq 3 | awk '!(NR == 2)'
1
3
$ seq 3 | awk '/2/'
2
# /2/ 는 $0 ~ /2/ 와 같으므로 !/2/ 는 결과적으로 !($0 ~ /2/) 와 같은 식이다.
$ seq 3 | awk '!/2/'
1
3
$ seq 3 | awk '!($0 ~ /2/)'
1
3
$ seq 3 | awk '$0 !~ /2/'
1
3
```

`sed`와 같이 `awk`에서도 `,`를 이용한 range를 사용할 수 있다.

```bash
$ seq 10 | awk 'NR == 3, NR ==3'
3
4
5
# BEGIN이 포함되는 라인부터 END가 포함되는 라인까지 출력
$ awk '/BEGIN/, /END/' a.txt
# BEGIN이 포함되는 라인부터, 공백 라인까지 출력
$ awk '/BEGIN/,/^$/' a.txt
# BEGIN이 포함되는 라인부터 끝까지 출력
$ awk '/BEGIN/,0' a.txt
$ awk '$1 ~ /BEGIN/, $2 ~ /END/' a.txt
$ awk 'NR == 5, /END/' a.txt
```

`&&`, `||` 와 같은 logical operator를 사용할 수 있다. 

```bash
$ cat file
111
222 foo
333
444
555
666
777
888 foo
999
$ awk '(NR < 4) || (NR > 6) && /foo/' a.txt
111
222 foo
333
888 foo
$ awk '((NR < 4) || (NR > 6)) && /foo/' a.txt
222 foo
888 foo
```

`print`, `printf`는 function이 아니고 statement이다. 따라서
`()`를 사용하지 않아도 된다.

```bash
$ cat a.txt
Amelia     555-5553
Anthony    555-3412
Becky      555-7685
$ awk '{ printf "%-10s %s\n", $1, $2 } a.txt
Amelia     555-5553
Anthony    555-3412
Becky      555-7685
$ awk 'BEGIN { format = "%-10s %s\n"
               printf format, "Name", "Phone"
               printf format, "-----", "------" }
             { printf format, $1, $2}' a.txt
Name       Phone
-----      ------
Amelia     555-5553
Anthony    555-3412
Becky      555-7685
$ awk 'BEGIN { var = "bar"; printf "foo" var "\n" }'
```

`print` statement는 OFS (output field seperator), ORS (output record
seperator)를 사용한다. 기본적인 OFS는 space이고 ORS는 newline이다.

```bash
$ cat a.txt
AAA,BBB,CCC,III
XXX,YYY,MMM,OOO
ZZZ,CCC,DDD,KKK
$ awk '{ print $1 $2 $3 }' FS=, a.txt
AAABBBCCC
XXXYYYMMM
ZZZCCCDDD
$ awk '{ print $1, $2, $3 }' FS=, a.txt
AAA BBB CCC
XXX YYY MMM
ZZZ CCC DDD
$ awk '{ print $1, $2, $3 }' FS=, OFS=: a.txt
AAA:BBB:CCC
XXX:YYY:MMM
ZZZ:CCC:DDD
$ awk '{ print $1, $2, $3 }' FS=, OFS=: ORS=@ a.txt
AAA:BBB:CCC@XXX:YYY:MMM@ZZZ:CCC:DDD@
# print statement에 parameter가 없다면 $0 값이 출력된다.
$ awk '{ print }' a.txt
AAA,BBB,CCC,III
XXX,YYY,MMM,OOO
ZZZ,CCC,DDD,KKK
```

`exit`를 이용하면 실행을 종료할 수 있다. END블록이 있으면 실행된다.

```bash
$ awk 'BEGIN { exit 1 } END { print "END..." }'
END...
$ echo $?
1
$ awk 'BEGIN { exit }'; echo $?
0
```

## Data Types

데이터타입은 숫자와 문자열이 존재한다. 사용된 곳에 따라 숫자가 문자열로 문자열이 숫자로
취급될 수 있다.

```bash
$ awk 'BEGIN{ print ( 12 + 3 ) }'
15
$ awk 'BEGIN{ print ( "12" + 3 ) }'
15
$ awk 'BEGIN{ print ( "12" + "3" ) }'
15
$ awk 'BEGIN{ print ( substr(12345, 1, 2) ) }'
12
$ awk 'BEGIN{ print ( substr(12345, 1, 2) + 3) }'
15
```

`awk`에서 single quotes는 사용하지 않는다. 문자열에 대해 double
quotes만 사용한다. 문자열은 기본적으로 escape sequence가 처리된다.

```bash
$ awk -f - <<\EOF
BEGIN {
  print "hello"
  print 'world' # error
}
EOF
$ printf %s '111\222:333\444' | awk -F: '$1 == "111\222" { print $1 }'
$ printf %s '111\222:333\444' | awk -F: '$1 == "111\\222" { print $1 }'
111\222
# 문자열의 접두사가 숫자인 경우 숫자로 취급될 때 접두사만 숫자로 변환된다.
$ awk 'BEGIN{ print ("12abc" + 3) }'
```

`" "`과 같은 공백은 문자열을 이어 붙이는 효과가 있다.

```bash
$ awk 'BEGIN{ AA=11     22   33; print AA}'
112233
$ awk 'BEGIN{ AA=11 22 33; print "Number is: "     AA }'
Number is: 112233
$ awk 'BEGIN{ AA=11 " " 22 "   " 33; print AA }'
11 22     33
$ awk 'BEGIN{ two = 2; three = 3; print ( two three ) + 4 }'
27
# 11 과 1 이 이어붙어져서 문자열이 되고 12와 "111"은 문자열 비교를 하기 때문에 결과는 참이다.
$ awk 'BEGIN{ print (12 > 11 1) }'
1
$ awk 'BEGIN{ print (12 > 111) }'
0
```

8진수 16진수 scientific notation 모두 사용할 수 있다.

```
11       (10 진수)
011      (8  진수, 10 진수로 9)
0x11     (16 진수, 10 진수로 17)

+3.14    (+, - 기호의 사용)
-3.14

105
1.05e+2  (scientific notation)
1050e-1
```

roundmode는 기본적으로 "N" (roundTiesToEven) 이다.

| Rounding Mode                         | IEEE Name           | ROUNDMODE  |
|---------------------------------------|---------------------|------------|
| Round to nearest, ties to even        | roundTiesToEven     | "N" or "n" |
| Round toward plus Infinity            | roundTowardPositive | "U" or "u" |
| Round toward negative Infinity        | roundTowardNegative | "D" or "d" |
| Round toward zero                     | roundTowardZero     | "Z" or "z" |
| Round to nearest, ties away from zero | roundTiesToAway     | "A" or "a" |

```bash
$ awk '                    
BEGIN {
    x = -4.5
    for (i = 1; i < 10; i++) {
        x += 1.0
        printf("%4.1f => %2.0f\n", x, x)
    }
}'
-3.5 => -4
-2.5 => -2
-1.5 => -2
-0.5 => -0
 0.5 =>  0
 1.5 =>  2
 2.5 =>  2
 3.5 =>  4
 4.5 =>  4
# "A" ( roundTiesToAway )
$ awk -M -v ROUNDMODE=A '
BEGIN {
    x = -4.5
    for (i = 1; i < 10; i++) {
        x += 1.0
        printf("%4.1f => %2.0f\n", x, x)
    }
}' 
-3.5 => -4
-2.5 => -3
-1.5 => -2
-0.5 => -1
 0.5 =>  1
 1.5 =>  2
 2.5 =>  3
 3.5 =>  4
 4.5 =>  5 
```

문자열을 숫자로 만들려면 `0`을 더해주면 된다.

```
$ awk 'BEGIN{ print (12 > "111"+0) }
0
$ awk 'BEGIN{ print (substr(12345,1,2)+0 > 111) }'
0
```

`print` statement를 이용하여 출력할 때 `OFMT`가 적용되고 숫자에서 문자열로
변환될 때는 `CONVFMT`가 적용된다. 둘다 기본 값은 `%.6g`이다. 문자열이
입력되는 경우는 `OFMT`가 적용되지 않고 있는 그대로 출력된다. 만약 문자열이
산술연산 되는 경우는 숫자로 변환되어 출력되기 때문에 `CONVFMT`와 `OFMT`가
적용된다.

```bash
$ awk 'BEGIN{ print OFMT; print CONVFMT }'
%.6g
%.6g
$ echo "1.23456789" | awk '{ 
    OFMT = "%.2f"    
    print $1        # $1 == "1.23456789"  입력값은 기본적으로 스트링
    a = $1 + 0      # a == 1.23456789     산술연산에 의해 숫자로 변경됨
    print a         # a 는 숫자이므로 OFMT 값의 적용을 받음
    b = a ""        # b == "1.23457"  b 는 CONVFMT 값에 따라 스트링이 됨
    print b         # 스트링을 프린트할 때는 OFMT 값의 적용을 받지 않음
}' 
1.23456789
1.23
1.23457
```

## Variables

파일별로 변수를 넘겨 보자.

```bash
$ awk '{ print AA, $0 }' AA=111 a.txt AA=222 b.txt
111 file1 record ...
111 file1 record ...
111 file1 record ...
222 file2 record ...
222 file2 record ...
222 file2 record ...
$ awk '{ ... }' RS='#' FS=',' a.txt RS='@' FS=':' b.txt
```

`BEGIN` 블록은 레코드를 읽어들여 명령 사이클을 시작하기 전에 실행되기
때문에 `BEGIN` 블록에 변수를 적용하려면 `-v` 옵션을 사용하자.

```bash
$ awk 'BEGIN{ print AA, BB }' AA=11 B=22
$
$ awk -v AA=11 -v BB=22 'BEGIN{ print AA, BB }'
11 22
```

`FS` 변수는 `-F`옵션을 이용해도 된다.

```bash
$ awk -F, 'BEGIN{ print "FS is: " FS }'
FS is: ,
% awk -F ',|:' 'BEGIN{ print "FS is: " FS }'
FS is: ,|:
```

`awk`에 shell variable을 전달할 때 공백을 포함하는 경우 double
quotes를 사용해야 한다.

```bash
$ MYVAR="foo bar"
$ awk -v AA=$MYVAR 'BEGIN{ ... }' # error
$ awk -v AA="$MYVAR" 'BEGIN{ ... }'
$ awk -v "AA=$MYVAR" 'BEGIN{ ... ";
```

심볼 테이블은 `SYMTAB`을 이용하여 접근하자.

```bash
$ awk '
BEGIN {
    foo = 100
    SYMTAB["foo"] = 200
    print foo

    bar[2] = 300
    SYMTAB["bar"][2] = 400
    print bar[2]
}'
200
400

$ awk '
BEGIN {
    answer = 10.5
    multiply("answer", 4)
    print "The answer is", answer
}
function multiply(var, amount) {
    return SYMTAB[var] *= amount
}'
The answer is 42
```

## Regexp

`/`와 `/`에 regexp를 표현할 때 이것을 regexp contstant라고 한다.

```bash
$ printf '111\t222:333\t444' | awk -F: '$1 ~ /11\t22/ { print $1 }'
111    222
# '~' 이 없으면 '$0' 와 매칭한다.
# awk '$0 ~ /^foobar$/ ...'
$ echo 'foobar' | awk '/^foobar$/ { print }'
foobar
$ echo 'foobar' | awk '{ if ( /^foobar$/ ) print }'
foobar
$ echo 'foobar' | awk '{ AA = /^foobar$/;  print AA }'
1
$ echo 'foobar' | awk '{ AA = ( $0 ~ /^foobar$/ );  print AA }'
1
$ echo 'foobar' | awk '{ AA = /^fooxxx$/;  print AA }'
0
```

문자열로 regexp를 이용할때 `~`는 생략할 수 없다.

```bash
$ echo 'foobar' | awk '{ var = "ob"; if ( $0 ~ "^fo" var "ar$" ) print }'
foobar
$ echo 'foobar' | awk '{ regex = "^foobar$"; if ( $0 ~ regex ) print }'
foobar
# regexp constant
$ printf %s '"111\222":"333\444"' | awk -F: '$1 ~ /"111\\222"/ { print $1 }'
"111\222"
# regexp string
$ printf %s '"111\222":"333\444"' | awk -F: '$1 ~ "\"111\\\\222\"" { print $1 }'
"111\222"
```

`awk`에서 주로 사용하는 regexp extension은 다음과 같다.

| regexp    | extension           |
| ------    | ---------           |
| `\s`      | `[[:space:]]`       |
| `\S`      | `[^[:space:]]`      |
| `\w`      | `[[:alnum:]_]`      |
| `\W`      | `[^[:alnum:]_]`     |
| `\y` `\B` | word boundary       |
| `\<`      | word boundary start |
| `\>`      | word boundary end   |

```bash
$ echo 'abc %-= def.' | awk '{ gsub(/\y/,"X") }1'
XabcX %-= XdefX.
$ echo 'abc %-= def.' | awk '{ gsub(/\B/,"X") }1'
aXbXc X%X-X=X dXeXf.X
$ echo 'abc %-= def.' | awk '{ gsub(/\</,"X") }1'
Xabc %-= Xdef.
$ echo 'abc %-= def.' | awk '{ gsub(/\>/,"X") }1'
abcX %-= defX.
```

대소문자 구분없이 매칭하는 방법은 다음과 같다.

```bash
$1 ~ /[Ff]oo/ { ... }
tolower($1) ~ /foo/ { ... }
x = "aB"
if (x ~ /ab/) ...   # fail
IGNORECASE = 1
if (x ~ /ab/) ...   # success
```

## Functions

`awk`의 함수에서 scalar argument는 pass by value이고 array argument는
pass by reference이다.

```bash
$ awk 'BEGIN { v = 100; f(v); print v } function f(p) { p = 200 }'
100
$ awk 'BEGIN { v[1] = 100; f(v); print v[1] } function f(p) { p[1] = 200 }'
200
```

function은 `BEGIN`, `END`, `{}` 블록에서 정의할 수 없다. function
name은 반드시 `(`와 붙어있어야 한다.

```bash
BEGIN {
  FS = "\t";
}
{
  log_level = $2;
}
(log_level == "ERROR") {
  print red($0)
}
(log_level == "WARN") {
  print yellow($0)
}
END {
}
function red(str) {
  return "\033[1;31m" str "\033[0m"
}
function yellow(s) {
  return "\033[1;33m" str "\033[0m"
}
```

function에서 사용할 local variable은 특이하게 function argument에
추가합니다.

```bash
function name(a)
function name(  i)
function name(a,    i)
function name(a, b, c
                 i, j, k)
```

`@`를 사용하면 indirection call할 수 있다.

```bash
$ awk 'BEGIN { 
    AA = "foo"; @AA(11) 
    AA = "bar"; @AA(22,33)
} 
function foo(n1)     { print "Im foo :", n1 }
function bar(n1, n2) { print "Im bar :", n1 ,n2 }
' 
Im foo : 11
Im bar : 22 33
```

## BEGIN, END

`BEGIN`, `BEGINFILE`, `ENDFILE`, `END` block은 다음과 같은 순서로
실행된다.

```bash
$ awk '
BEGIN { print "BEGIN " FILENAME } 
    BEGINFILE { print "BEGINFILE " FILENAME } 
        { print $0 " .............FNR: " FNR ", NR: " NR }
    ENDFILE { print "ENDFILE " FILENAME } 
END { print "END " FILENAME }
' a.txt b.txt c.txt

BEGIN            # FILENAME is empty.
BEGINFILE a.txt
111 .............FNR: 1, NR: 1      # 111, 222, 333 은 파일 내용이다.
222 .............FNR: 2, NR: 2
333 .............FNR: 3, NR: 3
ENDFILE a.txt
BEGINFILE b.txt
444 .............FNR: 1, NR: 4
555 .............FNR: 2, NR: 5
666 .............FNR: 3, NR: 6
ENDFILE b.txt
BEGINFILE c.txt
777 .............FNR: 1, NR: 7
888 .............FNR: 2, NR: 8
999 .............FNR: 3, NR: 9
ENDFILE c.txt
END c.txt
```

## Builtin Variables

### RECORD

* `NR`
* `FNR`
* `RS`
* `ORS`
* RT
  * Record Terminator

### FIELD

* NF
  * Number of fields
* FS
  * Input field separator
* OFS
* FIELDWIDTHS
* FPAT

### FORMAT

* CONVFMT
* OFMT
* PREC
* ROUNDMODE

### MATCHING

* IGNORECASE
* RSTART
* RLENGTH

### ARGUMENT

* ARGC
* ARGV
* ARGIND

### ARRAY

* SUBSEP

### ETC

* BINMODE
* ENVIRON
* ERRNO
* FILENAME
* FUNCTAB
* PROCINFO
* TEXTDOMAIN

## Record seperation

RS (Record seperator)와 RT(Record terminator)를 이용하여 분리하자.

```bash
$ echo '11 X 22 XX 33 XXX 44 X+ 55 XX 66 XXX 77 Y* 88 XX 99' |
awk '
{ printf "(%s) RT : \"%s\"\n", $0, RT } 
END{ print "===========\nNR : " NR}
' RS='XXX'      

(11 X 22 XX 33 ) RT : "XXX"
( 44 X+ 55 XX 66 ) RT : "XXX"
( 77 Y* 88 XX 99
) RT : ""
===========
NR : 3
```

## Field seperation

## `$0=$0, $1=$1`

## Redirection

## getline

## Arrays

## Control statements

* next

* nextfile

* if-else

```bash
if (x % 2 == 0)
    print "x is even"
else
    print "x is odd"
```

* while

```bash
i = 1
while (i <= 10) {
    print $i
    i++
}
```

* do-while

```bash
i = 1
do {
    print $i
    i++
} while (i <= 10)
```

* for

```bash
for (i = 1; i <= 10; i++)
    print $i
........................

for (idx in arr) {
    printf "index: %s, value: %s\n" ,idx ,arr[idx]
}
```

* switch

```bash
$ awk '{ 
    switch ($2) {
        case "+" :
            r = $1 + $3
            break
        case "-" :
            r = $1 - $3
            break
        case "*" :
            r = $1 * $3
            break
        case "/" :
            r = $1 / $3
            break
        default :
            print "Error"
            next
    }
    print r
}'

1 + 2
3
2 * 3
6
5 -2
Error
5 - 2
3
12 / 3
4
```

* break
* continue
* exit


## Operators

| Operators              | Description                                  |
| ----------             | ------------                                 |
| `++ --`                | Increment and decrement (prefix and postfix) |
| `^`                    | Power                                        |
| `* / %`                | Multiply, divide and modulus (remainder)     |
| `+ -`                  | Add and subtract                             |
| `nothing`              | String concatenation                         |
| `> >= < <= == != ~ !~` | Relational operators                         |
| `!`                    | Negate expression                            |
| `&& ||`                | Logical AND OR                               |
| `= += -= *= /= %= ^=`  | Assignment                                   |

## Builtin Functions

### Numeric Functions
### String Functions
### I/O Functions
### Time Functions
### Bitweise Functions
### Type Functions
### I18N Functions

## TCP/IP

다음은 웹서버를 구현한 것이다. webserver.awk와 cat.jpg를 복사하고
`awk -f webserver.awk`를 실행하자.

```bash
@load "filefuncs"
@load "readfile"
BEGIN {
    RS = ORS = "\r\n"
    datefmt = "Date: %a, %d %b %Y %H:%M:%S %Z"
    socket = "/inet/tcp/8080/0/0"
    while ((socket |& getline) > 0) {
        if ( $1 == "GET" ) { 
            print "GET : ", $2 
            $2 = substr($2,2)
            switch ($2) {
                case /\.(jpg|gif|png)$/ :
                    sendImage($2)
                    break
                default :
                    sendHtml()
            }
        }
    }
}
function sendHtml(    a, arr) {
    arr["type"] = "text/html"
    arr["content"] = "\
     <HTML>\n\
     <HEAD><TITLE>Out Of Service</TITLE></HEAD>\n\
     <BODY>\n\
           <H1>This site is temporarily out of service.</H1>\n\
           <P><img src=cat.jpg></P>\n\
     </BODY>\n\
     </HTML>\n\
     "
    arr["length"] = length(arr["content"])
    arr["date"] = strftime(datefmt, systime(), 1)

    send(arr)
}
function sendImage(file,     c, a, arr, type) {
    RS="\n"
    c = "file -b --mime-type '" file "'"
    c | getline type; close(c)
    RS="\r\n"
    arr["type"] = type
    stat(file, a)
    arr["length"] = a["size"]
    arr["content"] = readfile(file)
    arr["date"] = strftime(datefmt, systime(), 1)

    send(arr)
}
function send(arr) {
    print "HTTP/1.0 200 OK"                    |& socket
    print arr["date"]                          |& socket
    print "Server: AWK"                        |& socket
    print "Content-Length: " arr["length"]     |& socket
    print "Content-Type: " arr["type"]         |& socket
    print ""                                   |& socket
    print arr["content"]                       |& socket

    print "close socket"
    close(socket)
}
```

다음은 chatting server를 구현한 것이다.

```bash
$ awk '
@load "fork"
BEGIN{
    socket = "/inet/tcp/8080/0/0"
    socket |& getline
    if (( pid = fork() ) == 0 ) {
        while ((getline msg < "/dev/stdin") > 0 )   # child
            print msg |& socket
    } else {
        while ((socket |& getline msg) > 0 )        # parent
            print msg
        system("kill " pid)
    }
}'
```

다음은 chatting client를 구현한 것이다.

```bash
$ awk '
@load "fork"
BEGIN{
    socket = "/inet/tcp/0/some.host.com/8080"
    print "" |& socket
    if (( pid = fork() ) == 0 ) {
        while ((getline msg < "/dev/stdin") > 0 )   # child
            print msg |& socket
    } else {
        while ((socket |& getline msg) > 0 )        # parent
            print msg
        system("kill " pid)
    }
}'
```

## Debugging

다음과 같이 `getline`, `print`에 오타가 있는 경우도 문제 없이 실행된다.

```bash
$ awk 'BEGIN { getine var < "file1"; prit var }' 
```

`--lint`옵션을 사용하면 warning message를 볼 수 있다.

```bash
$ awk --lint 'BEGIN { getine var < "file1"; prit var}' 
awk: cmd. line:1: warning: statement may have no effect
awk: warning: statement may have no effect
awk: cmd. line:1: warning: reference to uninitialized variable 'getine'
awk: cmd. line:1: warning: reference to uninitialized variable 'var'
awk: cmd. line:1: warning: reference to uninitialized variable 'prit'
awk: cmd. line:1: warning: reference to uninitialized variable 'var'
```

`awk`는 실행파일이 1M도 안되지만 debugger가 내장되어 있다. debugger를
사용하려면 `-D`옵션을 이용해야 한다. 그리고 별도의 파일만 debugging이
가능하다.

```bash
$ cat transpose.awk 
{
    for(i=1; i<=NF; i++)
        r[i]=r[i] sep $i
    sep=FS
} 
END {
    for(i=1; i<=NF; i++) 
        print r[i]
}
..................................
$ cat matrix 
a1;a2;a3;a4;a5
b1;b2;b3;b4;b5
c1;c2;c3;c4;c5
d1;d2;d3;d4;d5
..................................
$ awk -F\; -f transpose.awk matrix
a1;b1;c1;d1
a2;b2;c2;d2
a3;b3;c3;d3
a4;b4;c4;d4
a5;b5;c5;d5
..................................
# '-D' 를 사용하면 디버깅을 할 수 있다.
$ awk -D -F\; -f transpose.awk matrix
gawk> ....
```

# Advanced

## 필드 출력하기

* 3번째 필드를 출력하자.

```bash
awk '{print $3}' a.txt
```

* 1, 5, 6번째 필드를 출력하자.

```bash
awk '{print $1 $5 $6}' a.txt
```
