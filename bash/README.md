<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [References](#references)
- [Shell Metachars](#shell-metachars)
- [Basic Usages](#basic-usages)
- [Shell Operation](#shell-operation)
- [Shell Parameters](#shell-parameters)
    - [Positional Parameters](#positional-parameters)
    - [Special Parameters](#special-parameters)
- [Shell Expansions](#shell-expansions)
    - [brace expansion](#brace-expansion)
        - [string lists](#string-lists)
        - [ranges](#ranges)
        - [combining and nesting](#combining-and-nesting)
        - [ranges with variables](#ranges-with-variables)
        - [prevent brace expansion](#prevent-brace-expansion)
    - [tilde expansion](#tilde-expansion)
    - [parameter and variable expansion](#parameter-and-variable-expansion)
        - [basic usage](#basic-usage)
        - [string length](#string-length)
        - [substring removal](#substring-removal)
        - [use a default value](#use-a-default-value)
        - [assign a default value](#assign-a-default-value)
        - [use an alternate value](#use-an-alternate-value)
        - [display error if null or unset](#display-error-if-null-or-unset)
        - [substring expansion](#substring-expansion)
        - [search and replace](#search-and-replace)
        - [case modification](#case-modification)
        - [indirection](#indirection)
    - [command substitution](#command-substitution)
    - [arithmetic expansion](#arithmetic-expansion)
    - [process substitution](#process-substitution)
    - [word splitting](#word-splitting)
    - [filename expansion](#filename-expansion)
    - [quote removal](#quote-removal)
- [Shell Builtin Commands](#shell-builtin-commands)
    - [Bourne Shell Builtins](#bourne-shell-builtins)
    - [Bash Builtins](#bash-builtins)
- [Shell Variables](#shell-variables)
    - [Bourne Shell Variables](#bourne-shell-variables)
    - [Bash Variables](#bash-variables)
- [Bash Features](#bash-features)
    - [Bash Startup Files](#bash-startup-files)
        - [login shell](#login-shell)
        - [non-login shell](#non-login-shell)
    - [Bash Conditional Expressions](#bash-conditional-expressions)
    - [Shell Arithmetic](#shell-arithmetic)
    - [Arrays](#arrays)
    - [Controlling the Prompt](#controlling-the-prompt)
- [Job Control](#job-control)
    - [Job Control Builtins](#job-control-builtins)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

bash에 대해 정리한다.

# References

* [Bash script](https://www.gitbook.com/book/mug896/shell-script/details)
  * 친절한 한글
* [bash reference manual @ gnu](https://www.gnu.org/software/bash/manual/bash.html)
* [Advanced Bash-Scripting Guide](http://www.tldp.org/LDP/abs/html/index.html)

# Shell Metachars

```
 ( )   `   |   &   ;               # command substitution
 &&  ||                            # AND, OR 
 <   >   >>                        # redirection 
 *   ?   [ ]                       # glob 
 "   '                             # quote 
 \   $      
 =   +=                            
```

# Basic Usages

```bash
# file name can be anything except NUL, / on linux
echo "hello world" > [[]].txt

# '-' can be stdin for input
echo hello world | cat -
# '-' can be stdout for output
edcho hello world | cat a.txt -
seq 10 | paste - - -

# > 
# redirect the stdout of a command to a file. 
# Overwrite the file if it already exists.
echo "foo" > a.txt

# <
# redirect file contents to stdin of command.
cat < a.txt

# >>
# redirect and append stdout of a command to a file
echo "foo" >> a.txt

# 2>
# redirect the stderr of a command to a file
echo "foo" 2> a.txt

# 2>>
# redirect and append the stderr of a command to a file
echo "foo" 2>> a.txt

# &>
# redirect stdout, stderr of a command to a file
echo "foo" 2>> a.txt

# 1>&2
# redirect stdout of a command to stderr
foo=$(echo "foo" 1>&2)
echo $foo

# 2>&1
# redirect stderr of a command to stdout
foo > /dev/null 2>&1

# |
# redirect stdout of first command to stdin of other command
ls -al | grep foo

# $
foo="hello world"
echo $foo

# ``
# command substitution
echo `date`

# $()
# The alternative form of command substitution
echo $(date)

# &&, ||
# execute several commands
make && make install
echo "hello" || echo "world"

# ;
# execute several commands on a line
false; echo "foo"

# ''
# full quoting (single quoting)
#  'STRING' preserves all special characters within STRING. 
# This is a stronger form of quoting than "STRING".
echo '$USER'

# ""
# partial quoting (double quoting)
# "STRING" preserves (from interpretation) 
# most of the special characters within STRING.
# except $, `, \
echo "foo"
echo "$USER"
echo "Now is $(date)"
echo "Now is `date`"

# "''"
bash -c "/bin/echo foo 'bar'"

# \" \$foo
bash -c "/bin/echo '{ \"user\" : \"$USER\" }'"
echo "\$foo \" \`"

# $
# variable substitution
msg="bar"
echo "foo $msg"

# ${}
# parameter substitution
msg="bar"
echo "foo ${msg}"
foo=
foo=${foo-"aaa"}
echo $foo
bar=
bar=${bar:-"aaa"}
echo $bar

# \
# represent a command in several lines
echo "foo"
echo \
  "foo"
  
# {1..10}
# 
echo {1..10}

# {string1,string2}
#
cp {foo.txt,bar.txt} a/

# looping constructs
# until, while, for, break, continue
until test-commands; do consequent-commands; done
while test-commands; do consequent-commands; done
for name [ [in [words …] ] ; ] do commands; done
for (( expr1 ; expr2 ; expr3 )) ; do commands ; done

# while
while :
do
    echo "foo";
    sleep 1;
done

# for
for i in $(l)
do 
    echo $i
done

for (( i=0; i < 10; i++ ))
do
    echo $i
done

NUM=(1 2 3)
for i in ${NUM[@]}
do
    echo $i
done

# conditional consructs :
#   if, case, select, ((...)), [[...]]

# if
# comparison of numbers
# -eq, -ne, -gt, -ge, -lt, -le
# comparison of strings
# =(==), !=, -z(null), -n(not null)
if [ $a -eq $b ]; then
    echo $a
fi

# grouping commands : (), {}
# (command list) : command list는 subshell환경에서 실행된다.
( while true; do echo "hello"; sleep 1; done )
# {command list} : command list는 같은 shell환경에서 실행된다.
{ while true; do echo "hello"; sleep 1; done }

# variable은 unset, null, not-null과 같이 3가지 상태를 갖는다.
# unset state
declare A
local A
# null state
A=
A=""
A=''
# not-null state
A=" "
A="hello"
A=123

# function은 command들을 그룹화 하는 방법이다. 그룹의 이름을 사용하면 그룹안의 commands를 실행할 수 있다.
# name () compound-command [ redirections ]
H() { echo "hello"; }; H; echo "world";
# function name [()] compound-command [ redirections ]
function H() { echo "hello"; }; H; echo "world";

# '*', '?', '[]' 과 같은 glob characters 을 이용하여 filename, case, parameter expansion, [[]] expression 등에서 pattern matching이 가능하다.
# filename
ls *.[ch]
# [[]] expression
[[ $A = *.tar.gz ]]; echo $?
[[ $A = *dog\ cat* ]]; echo $?

# shell file의 첫줄에서 스크립트를 실행할 command line을 shabang line이라고 한다. 옵션은 하나로 제한된다.
#! /bin/bash
#! /bin/sed -f
#! /usr/bin/awk -f
#! /usr/bin/perl -T

# <<<
# redirect a string to stdin of a command
cat <<< "I am $USER"

# <<EOF EOF
# redirect several lines to stdin of a command
cat > a.txt <<EOF
I am the man.
HOST is $(hostname)
USER is $(USER)
EOF

# export
export foo=bar

# printf
Message1="foo"
Message2="bar
printf "%s %s \n" $Message1 $Message2

# sed
sed -i "s/foo/bar/g" a.txt

```

# Shell Operation

shell이 command를 읽고 실행하는 과정은 다음과 같다.

* commands 및 arguments를 읽어 들인다.
* 읽어 들인 commands를 Quoting과 동시에 words와 operators로 쪼갠다. 쪼개진 토큰들은 metacharaters를 구분자로 한다. 이때 alias expansion이 수행된다.
* 토큰들을 읽어서 simple, compound commands를 구성한다.
* 다양한 shell expansion이 수행되고 expanded tokens는 파일이름, 커맨드, 인자로 구성된다.
* redirection을 수행하고 redirection operators와 operands를 argument list에서 제거한다.
* command를 실행한다.
* 필요한 경우 command가 실행 완료될 때까지 기다리고 exit status를 수집한다.

# Shell Parameters

## Positional Parameters

`${N}`처럼 표기한다. N는 paramter의 서수이다.

a.sh `echo ${0}, ${1}, ${2}`

```bash
echo a.sh "hello" "world" "foo"
```

## Special Parameters

* `$*`
  * 모든 parameter들
  * a.sh `echo $*` `bash a.sh 1 2 3 4 5`
    * 결과는 `1 2 3 4 5`

* `$@`
  * `$*`와 같다.

* `$#`
  * 마지막 parameter
  * a.sh `echo $#` `bash a.sh 1 2 3 4 5`
    * 결과는 `5`

* `$?`
  * 마지막 실행된 foreground pipeline의 exit status
  * a.sh `echo $?` `bash a.sh 1 2 3 4 5`
    * 결과는 `0`

* `$-`
  * 현재 설정된 option flags
  * a.sh `echo $-` `bash a.sh 1 2 3 4 5` 
    * 결과는 `hB`
  
* `$$`
  * 실행 프로세스 ID
  * a.sh `echo $$` `bash a.sh 1 2 3 4 5`
    * 결과는 `3`

* `$!`
  * 가장 최근에 실행된 background 프로세스 ID
  * a.sh `echo $!` `bash a.sh 1 2 3 4 5`
    * 결과는 ``

* `$0`
  * shell or shell script
  * a.sh `echo $!` `bash a.sh 1 2 3 4 5`
    * 결과는 `a.sh`

* `$_`
  * shell 혹은 shell script의 절대경로
  * a.sh `echo $_` `bash a.sh 1 2 3 4 5`
    * 결과는 `/bin/bash`

# Shell Expansions

command line은 tokens로 나눠진 후 다음과 같은 순서 대로 expansion처리
된다. brace expansion, tilde expansion, parameter and variable
expansion, command expansion, arithmetic expansion, process
substitution, word splitting, filename expansion, quote removal

## brace expansion

### string lists

보통 다음과 같은 형식으로 사용한다.

```
{string1, string2,..., stringN}
```

`,`가 사용되지 않은 단일 항목은 확장되지 않는다. `,`전 후에 공백을
사용할 수는 없다. 문자열에 공백이 포함되는 경우 quote해야
expansion할 수 있다.

```bash
# ',' 가 사용되지 않은 단일 항목은 확장되지 않는다.
$ echo {hello}
{hello}

# ',' 전,후에 공백이 사용되면 확장이 되지 않는다.
$ echo X{apple, banana, orange, melon}Y
X{apple, banana, orange, melon}Y

# string 내에 공백이 포함될 경우 quote 한다.
$ echo X{apple,"ban ana",orange,melon}Y
XappleY Xban anaY XorangeY XmelonY
```

preamble과 postscript의 사용

```bash
$ echo X{apple,banana,orange,melon}         # 여기서 X 는 preamble
Xapple Xbanana Xorange Xmelon

$ echo {apple,banana,orange,melon}Y         # Y 는 postscript
appleY bananaY orangeY melonY

# preamble, postscript 이 없을 경우 단지 space 로 분리되어 표시됩니다.
$ echo {apple,banana,orange,melon}
apple banana orange melon

$ echo X{apple,banana,orange,melon}Y
XappleY XbananaY XorangeY XmelonY

# '/home/bash/test/' 가 preamble 에 해당
$ mkdir /home/bash/test/{foo,bar,baz,cat,dog}
$ ls /home/bash/test/
bar/  baz/  cat/  dog/  foo/

$ tar -czvf backup-`date +%m-%d-%Y-%H%M`.tar.gz \
     --exclude={./public_html/cache,./public_html/compiled,./public_html/images} \
     ./public_html

# 'http://docs.example.com/slides_part' 는 preamble '.html' 는 postscript 에 해당
$ curl --remote-name-all http://docs.example.com/slides_part{1,2,3,4}.html
$ ls
slides_part1.html slides_part2.html slides_part3.html slides_part4.htmlx
```

globbing과 사용

```bash
mv *.{png,gif,jpg} ~/tmp
```

null값과 사용

```bash
$ echo aa{,}11
aa11 aa11

$ echo aa{,,}11
aa11 aa11 aa11

$ echo b{,,,A}a
ba ba ba bAa

$ cp test.sh{,.bak}
$ ls
test.sh test.sh.bak

$ ls  shell/{,BB/}rc.d
shell/rc.d
...
shell/BB/rc.d
...

$ paste <( shuf -n5 -i 1-99 ){,}
57      79
59      68
75      90
53      72
85      98

$ paste <( shuf -n5 -i 1-99 ){,,} 
48      18      91
58      24      95
91      54      66
31      14      10
44      8       55
```

변수와 같이 사용

```bash
$ AA=hello

# 변수확장이 뒤에 일어나므로 {a,b}$AA 는 a$AA b$AA 와 같게 됩니다.
$ echo {a,b}$AA
ahello bhello

# 그냥 $AA 로 사용하면 X$AAY 와 같아지므로 ${AA} 나 "$AA" 를 사용합니다.
$ echo X{apple,${AA},orange,melon}Y
XappleY XhelloY XorangeY XmelonY
```

### ranges

보통 다음과 같은 형식으로 사용한다.

```
{< START >...< END > }
{< START >...< END >...< INCR >}
```

```
$ a=1 b=10

$ echo {$a..$b}         # no brace expansion
{1..10}

$ echo {5..12}
5 6 7 8 9 10 11 12

$ echo {c..k}
c d e f g h i j k

$ echo {5..k}          # 숫자와 알파벳을 섞어서 사용할수는 없습니다.
{5..k}

$ echo {1..10..2}      # 2씩 증가한다.
1 3 5 7 9
$ echo {10..1..2}
10 8 6 4 2

$ echo {a..z..3}
a d g j m p s v y
```

zero 패딩

```bash
$ echo {01..10}
01 02 03 04 05 06 07 08 09 10

$ echo {0001..5}
0001 0002 0003 0004 0005

# img001.png ~ img999.png 까지 생성
printf "%s\n" img{00{1..9},0{10..99},{100..999}}.png

# 01 ~ 10 까지 생성
$ for i in 0{1..9} 10; do echo $i; done
```

preamble 혹은 postscript의 사용

```bash
$ echo 1.{0..9}
1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9

$ echo __{A..E}__
__A__ __B__ __C__ __D__ __E__

$ echo {A..D}___{1..3}
A___1 A___2 A___3 B___1 B___2 B___3 C___1 C___2 C___3 D___1 D___2 D___3

# 'http://docs.example.com/slides_part' 는 preamble '.html' 는 postscript 에 해당
$ curl --remote-name-all http://docs.example.com/slides_part{1..4}.html
$ ls
slides_part1.html slides_part2.html slides_part3.html slides_part4.html
```

### combining and nesting

`{}`를 서로 붙이면 combining이 일어난다. `{}`안에서 `,`를
사용하면 nesting이 일어난다.

```bash
$ echo {A..Z}{0..9}
A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 C0 C1 C2 C3 C4 C5 C6
C7 C8 C9 D0 D1 D2 D3 D4 D5 D6 D7 D8 D9 E0 E1 E2 E3 E4 E5 E6 E7 E8 E9 F0 F1 F2 F3
F4 F5 F6 F7 F8 F9 G0 G1 G2 G3 G4 G5 G6 G7 G8 G9 H0 H1 H2 H3 H4 H5 H6 H7 H8 H9 I0
I1 I2 I3 I4 I5 I6 I7 I8 I9 J0 J1 J2 J3 J4 J5 J6 J7 J8 J9 K0 K1 K2 K3 K4 K5 K6 K7
K8 K9 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 M0 M1 M2 M3 M4 M5 M6 M7 M8 M9 N0 N1 N2 N3 N4
N5 N6 N7 N8 N9 O0 O1 O2 O3 O4 O5 O6 O7 O8 O9 P0 P1 P2 P3 P4 P5 P6 P7 P8 P9 Q0 Q1
Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 S0 S1 S2 S3 S4 S5 S6 S7 S8
S9 T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 U0 U1 U2 U3 U4 U5 U6 U7 U8 U9 V0 V1 V2 V3 V4 V5
V6 V7 V8 V9 W0 W1 W2 W3 W4 W5 W6 W7 W8 W9 X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 Y0 Y1 Y2
Y3 Y4 Y5 Y6 Y7 Y8 Y9 Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9

$ echo {{A..Z},{a..z}}
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s 
```

{ } 안에서 , 로 분리하여 nesting 을 할 수 있다.

```bash
$ echo {{A..Z},{a..z}}
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z
```

### ranges with variables

```bash
$ a=1 b=5
$ eval echo {$a..$b}    # eval 명령을 이용하면 원하는 결과를 얻을수 있습니다.
1 2 3 4 5

$ eval echo img{$a..$b}.png
img1.png img2.png img3.png img4.png img5.png
```

### prevent brace expansion

```bash
$ echo \{a..c}
{a..c}

$ echo {a..c\}
{a..c}

$ echo `{a..c}`
{a..c}
```

## tilde expansion

현재 디렉토리로 확장된다.

```
$ echo ~
/Users/iamslash

$ echo ~iamslash # iamslash 유저 디렉토리를 출력하라.
/Users/iamslash

$ echo ~+ # $PWD와 같다.

$ echo ~- # $OLDPWD와 같다.
```

## parameter and variable expansion


### basic usage

`{}`를 사용한다.

```bash
$ AA=cde

$ echo "ab$AAfg"       # 현재 변수 $AAfg 값은 null 이므로 'ab' 만 나온다. 
ab

$ echo "ab${AA}fg"     # '{ }' 을 이용하면 나머지 스트링과 구분할 수 있다.
abcdefg
```

### string length

```bash
${#PARAMETER}
```

```bash
$ AA="hello world"
$ echo ${#AA}
11

$ BB=( Arch Ubuntu Fedora Suse )
$ echo ${#BB[1]}     # [1] 번째 원소 Ubuntu 의 문자 수
6
$ echo ${#BB[@]}     # array BB 의 전체 원소 개수
4
```

### substring removal

```bash
${PARAMETER#PATTERN}
${PARAMETER##PATTERN}
${PARAMETER%PATTERN}
${PARAMETER%%PATTERN}
```

`#`는 앞에서 부터 `%`는 뒤에서 부터를 의미한다. 두개가 사용된 것은
greedy match를 의미한다.

```bash
AA="this.is.a.inventory.tar.gz"

$ echo ${AA#*.}               # 앞에서 부터 shortest match
is.a.inventory.tar.gz

$ echo ${AA##*.}              # 앞에서 부터 longest match
gz

$ echo ${AA%.*}               # 뒤에서 부터 shortest match
this.is.a.inventory.tar

$ echo ${AA%%.*}              # 뒤에서 부터 longest match
this

# 디렉토리를 포함한 파일명에서 디렉토리와 파일명을 분리하기
AA="/home/bash/bash_hackers.txt"

$ echo ${AA%/*}               # 디렉토리 부분 구하기
/home/bash

$ echo ${AA##*/}              # 파일명 부분 구하기
bash_hackers.txt
```

### use a default value

`-`는 4.3 BSD와 같이 예전 SHELL에서 `:-`를 처리하지 못하기 때문에 필요하다.

```bash
${PARAMETER:-WORD}
${PARAMETER-WORD}
```

```bash
$ AA=ubuntu

$ echo ${AA:-fedora}        # AA 에 값이 있으면 AA 값을 사용한다
ubuntu
$ echo ${AA-fedora}
ubuntu

$ unset AA

$ echo ${AA:-fedora}        # AA 가 unset 되어 없는 상태이므로  
fedora                      # 값은 fedora 가 된다.
$ echo ${AA-fedora}
fedora

$ AA=""

$ echo ${AA:-fedora}       # ':-' 는 null 은 값이 없는것으로 보고 fedora 를 사용한다.
fedora
$ echo ${AA-fedora}        # '-' 는 null 도 값으로 취급하기 때문에
                           # 아무것도 표시되지 않는다.


# $TMPDIR 변수값이 없으면 /tmp 디렉토리에서 myfile_* 을 삭제 
$ rm -f ${TMPDIR:-/tmp}/myfile_*

# WORD 부분에도 명령치환을 사용할 수 있다.
$ AA=${AA:-$(date +%Y)}

# FCEDIT 변수값이 없으면 EDITOR 변수값을 사용하고 EDITOR 변수값도 없으면 vi 를 사용
$ AA=${FCEDIT:-${EDITOR:-vi}}
```

array

```bash
$ AA=( 11 22 33 )

$ echo ${AA[@]:-44 55 66}    # AA 에 값이 있으면 AA 값을 사용한다
11 22 33
$ echo ${AA[@]-44 55 66}
11 22 33

$ AA=()                      # 또는 unset -v AA

$ echo ${AA[@]:-44 55 66}    # AA 가 unset 되어 존재하지 않는 상태이므로
44 55 66                     # 값은 44 55 66 이 된다.
$ echo ${AA[@]-44 55 66}
44 55 66

$ AA=("")

$ echo ${AA[@]:-44 55 66}    # ':-' 는 null 은 값이 없는것으로 보고 44 55 66 을 사용한다
44 55 66
$ echo ${AA[@]-44 55 66}     # '-' 는 null 도 값으로 취급하기 때문에
                             # 아무것도 표시되지 않는다.

$ AA=("" 77 88)

$ echo ${AA[@]:-44 55 66}
77 88
$ echo ${AA[@]-44 55 66}
77 88
```

### assign a default value

```bash
${PARAMETER:=WORD}
${PARAMETER=WORD}
```

값을 읽어오면서 대입도 하자.

```bash
AA=""

$ echo ${AA:-linux}
linux
$ echo $AA             # ':-' 는 AA 변수에 값을 대입하지 않는다.

$ echo ${AA:=linux}
linux
$ echo $AA             # ':=' 는 AA 변수에 값을 대입한다.
linux


# $TMPDIR 값이 없으면 /tmp 를 사용하고 TMPDIR 변수에 대입
$ cp myfile ${TMPDIR:=/tmp}

# $TMPDIR 변수가 설정된 상태이므로 cd 할 수 있다.
$ cd $TMPDIR
```

Array[@] 값은 대입되지 않는다.

```bash
$ AA=()

$ echo ${AA[@]:=11 22 33}
bash: AA[@]: bad array subscript
11 22 33
```

### use an alternate value

값이 있다면 대체 값을 사용하자.

```bash
${PARAMETER:+WORD}
${PARAMETER+WORD}
```

```bash
$ AA=hello

$ echo ${AA:+linux}       # AA 에 값이 있으므로 대체값 linux 를 사용한다.
linux
$ echo ${AA+linux}
linux

$ AA=""

$ echo ${AA:+linux}       # ':+' 는 null 은 값으로 취급하지 않기 때문에 null 을 리턴한다.

$ echo ${AA+linux}        # '+' 는 null 도 값으로 취급하기 때문에 대체값 linux 를 사용한다.
linux

$ unset AA

$ echo ${AA:+linux}       # 변수가 존재하지 않으므로 null 을 리턴한다.

$ echo ${AA+linux}


# 함수이름을 갖는 FUNCNAME 변수가 있을 경우 뒤에 '()' 를 붙여서 프린트 하고 싶다면
echo ${FUNCNAME:+${FUNCNAME}()}
```

### display error if null or unset

```bash
${PARAMETER:?[error message]}
${PARAMETER?[error message]}
```

값이 있다면 에러 메시지 내보내고 종료하자. 종료코드는 1(bash) 혹은 2(sh)이다.

```bash
$ AA=hello

$ echo ${AA:?null or not set}      # AA 에 값이 있으므로 AA 값을 사용한다
hello
$ echo ${AA?not set}
hello

$ AA=""

$ echo ${AA:?null or not set}       # ':?' 는 null 은 값으로 취급하지 않기 때문에
bash: AA: null or not set           # error message 를 출력하고 $? 값으로 1 을 리턴한다 
$ echo $?
1
$ echo ${AA?not set}                # '?' 는 null 도 값으로 취급하기 때문에 
                                    # 아무것도 표시되지 않는다.

$ unset AA

$ echo ${AA:?null or not set}       # 변수가 존재하지 않는 상태이므로 
bash: AA: null or not set           # 모두 error message 를 출력 후 종료한다.
$ echo $?
1
$ echo ${AA?not set}
bash: AA: not set
$ echo $?
1
$ echo ${AA?}                        # error message 는 생략할 수 있다.
bash: AA: parameter null or not set


# 예제
case ${AA:?"missing pattern; try '$0 --help' for help"} in
    (abc) ... ;;
    (*) ... ;;
esac
```

### substring expansion

```bash
${PARAMETER:OFFSET}
${PARAMETER:OFFSET:LENGTH}
```

```bash
AA="Ubuntu Linux"

$ echo "${AA:2}"
untu Linux
$ echo "${AA:2:4}"
untu
$ echo "${AA:(-5)}"
Linux
$ echo "${AA:(-5):2}"
Li
$ echo "${AA:(-5):-1}"
Linu

# 변수 AA 값이 %foobar% 일경우 % 문자 제거하기
$ echo "${AA:1:-1}"
foobar
```

array

```bash
$ ARR=(11 22 33 44 55)

$ echo ${ARR[@]:2}
33 44 55

$ echo ${ARR[@]:1:2}
22 33
```

positional parameters는 idx가 1부터 시작

```bash
$ set -- 11 22 33 44 55

$ echo ${@:3}
33 44 55

$ echo ${@:2:2}
22 33
```

### search and replace

```bash
${PARAMETER/PATTERN/STRING}
${PARAMETER//PATTERN/STRING}
${PARAMETER/PATTERN}
${PARAMETER//PATTERN}
```

```bash
$ AA="Arch Linux Ubuntu Linux Fedora Linux"

$ echo ${AA/Linux/Unix}               # Arch Linux 만 바뀌였다.
Arch Unix Ubuntu Linux Fedora Linux

$ echo ${AA//Linux/Unix}              # Linux 가 모두 Unix 로 바뀌였다
Arch Unix Ubuntu Unix Fedora Unix

$ echo ${AA/Linux}                    # 바꾸는 스트링을 주지 않으면 매칭되는 부분이 삭제된다.
Arch Ubuntu Linux Fedora Linux

$ echo ${AA//Linux}                  
Arch Ubuntu Fedora

-----------------------------------------
$ AA="Linux Ubuntu Linux Fedora Linux"

$ echo ${AA/#Linux/XXX}               # '#Linux' 는 맨 앞 단어를 의미
XXX Ubuntu Linux Fedora Linux

$ echo ${AA/%Linux/XXX}               # '%Linux' 는 맨 뒤 단어를 의미
Linux Ubuntu Linux Fedora XXX

$ AA=12345

$ echo ${AA/#/X}
X12345

$ echo ${AA/%/X}
12345X

$ AA=X12345X

$ echo ${AA/#?/}
12345X

$ echo ${AA/%?/}
X12345
```

`array[@]`는 원소별로 적용

```bash
$ AA=( "Arch Linux" "Ubuntu Linux" "Fedora Linux" )

$ echo ${AA[@]/u/X}                   # Ubuntu Linux 는 첫번째 'u' 만 바뀌었다
Arch LinXx UbXntu Linux Fedora LinXx

$ echo ${AA[@]//u/X}                  # 이제 모두 바뀌였다
Arch LinXx UbXntX LinXx Fedora LinXx
```

### case modification

```bash
${PARAMETER^}  # 단어의 첫 문자를 대문자로
${PARAMETER^^} # 단어의 모든 문자를 대문자로
${PARAMETER,}  # 단어의 첫 문자를 소문자로
${PARAMETER,,} # 단어의 모든 문자를 소문자로
```

```bash
$ AA=( "ubuntu" "fedora" "suse" )

$ echo ${AA[@]^}
Ubuntu Fedora Suse

$ echo ${AA[@]^^}
UBUNTU FEDORA SUSE

$ AA=( "UBUNTU" "FEDORA" "SUSE" )

$ echo ${AA[@],}
uBUNTU fEDORA sUSE

$ echo ${AA[@],,}
ubuntu fedora suse
```

### indirection

```bash
${!PARAMETER}
```

스크립트 실행중에 스트링으로 변수 이름을 만들어서 사용

```bash
$ hello=123

$ linux=hello

$ echo ${linux}
hello

$ echo ${!linux}    # '!linux' 부분이 'hello' 로 바뀐다고 생각하면 됩니다. 
123
```

sh의 경우 eval을 이용하여 indirection구현

```sh
$ hello=123

$ linux=hello

$ eval echo '$'$linux
123
----------------------

if test "$hello" -eq  "$(eval echo '$'$linux)"; then
...
fi
```

함수에 전달된 인수를 표시

```bash
---------- args.sh ------------
#!/bin/bash

for (( i = 0; i <= $#; i++ )) 
do
    echo \$$i : ${!i}              # ${$i} 이렇게 하면 안됩니다.
done

-------------------------------

$ args.sh 11 22 33
$0 : ./args.sh
$1 : 11                    
$2 : 22
$3 : 33
```

함수에 array 인자를 전달 할 때도 사용

```bash
#!/bin/bash

foo() {
    echo "$1"

    local ARR=( "${!2}" )      # '!2' 부분이 'AA[@]' 로 바뀐다.

    for v in "${ARR[@]}"; do
        echo "$v"
    done

    echo "$3"
}

AA=(22 33 44)
foo 11 'AA[@]' 55

################ output ###############

11
22
33
44
55
```

## command substitution

```bash
$( <COMMANDS> )
`<COMMANDS>`
```

command substitution은 subshell에서 실행된다. 실행결과에 해당하는 stdout
값이 pipe를 통해 전달된다. 일종의 IPC이다.

```bash
$ AA=$( pgrep -d, -x ibus )
$ echo $AA
17516,17529,17530,17538,17541

$ AA=`pgrep -d, -x ibus`
$ echo $AA
17516,17529,17530,17538,17541

$ cat /proc/`pgrep -x awk`/maps

$ cd /lib/modules/`uname -r`
```

backtick은 escape sequence가 다르게 처리된다.

```bash
# 원본 명령
$ grep -lr --include='*.md' ' \\t\\n'
filename_expansion.md
word_splitting.md

# 괄호형은 원본 명령 그대로 사용할 수 있다.
$ echo $( grep -lr --include='*.md' ' \\t\\n' )
filename_expansion.md word_splitting.md
--------------------------------------------------

# backtick 은 escape sequence 가 다르게 처리되어 값이 출력되지 않는다.
$ echo `grep -lr --include='*.md' ' \\t\\n'`
$
$ echo `grep -lr --include='*.md' ' \\\\t\\\\n'`
filename_expansion.md word_splitting.md
```

parent 프로세스의 변수값을 변경할 수 없다.

```bash
#!/bin/bash

index=30

change_index() {
  index=40
}

result=$(change_index; echo $index)

echo $result
echo $index

######### output ########

40
30
```

quotes가 중첩되도 좋다.

```bash
# 명령치환을 quote 하지 않은 경우
$ echo $( echo "
> I
> like
> winter     and     snow" )

I like winter and snow

# 명령치환을 quote 하여 공백과 라인개행이 유지되었다.
$ echo "$( echo "
> I
> like
> winter     and     snow" )"

I
like
winter     and     snow

---------------------------------

# quotes 이 여러번 중첩되어도 상관없다.

$ echo "$(echo "$(echo "$(date)")")"       
Thu Jul 23 18:34:33 KST 2015
```

null문자를 보낼 수 없다.

```bash
$ ls          # a, b, c  3개의 파일이 존재
a  b  c

# 파이프는 null 값이 정상적으로 전달된다.
$ find . -print0 | od -a
0000000   . nul   .   /   a nul   .   /   b nul   .   /   c nul

# 명령치환은 null 값이 모두 제거되었다.
$ echo -n "$(find . -print0)" | od -a
0000000   .   .   /   a   .   /   b   .   /   c
```

변수에 값을 대입할 때 마지막 newline들은 제거된다.

```bash
$ AA=$'hello\n\n\n'
$ echo -n "$AA" | od -a
0000000   h   e   l   l   o  nl  nl  nl    # 정상적으로 newline 이 표시된다.

$ AA=$(echo -en "hello\n\n\n")
$ echo -n "$AA" | od -a
0000000   h   e   l   l   o                # 명령치환은 newline 이 모두 제거된다.

---------------------------------------

$ cat file         # 파일의 마지막에 4 개의 newline 이 존재.
111
222



$ cat file | od -a
0000000   1   1   1  nl   2   2   2  nl  nl  nl  nl
0000013

$ AA=$(cat file)

$ echo -n "$AA" | od -a                   # 파일의 마지막 newline 이 모두 제거 되었다.
0000000   1   1   1  nl   2   2   2
0000007
```

`$( < filename )` 은 `$( cat filename )`과 같다.

```bash
$ cat file
11
44
55

$ echo $( < file )
11 44 55

$ echo $( cat file )
11 44 55
```

## arithmetic expansion

보통 다음과 같이 사용한다.

```bash
#(( arithmetic-expr ))
(( arithmetic-expr ))
```

산술연산을 계산한다.

```bash
$ cat > a.sh
echo $(( 3 + 7 ))
# bash a.sh
10
```

## process substitution

```bash
<( <COMMANDS> )
>( <COMMANDS> )
```

`command1 > >( command2 )` 명령의 경우 command1 의 stdout 이 command2 의
stdin 과 연결되며 `command1 < <( command2 )` 명령의 경우는 command2 의
stdout 이 command1 의 stdin 과 연결됩니다.  현재 shell pid 를 나타내는
$$ 변수는 subshell 에서도 동일한 값을 가지므로 >( ) 표현식 내에서의 FD
상태를 보기 위해서는 $BASHPID 변수를 이용해야 합니다.


```bash
# '>( )' 표현식 내의 명령은 subshell 에서 실행되므로 '$$' 값이 같게나온다.

$ { echo '$$' : $$ >&2 ;} > >( echo '$$' : $$ )
$$ : 504
$$ : 504

# 하지만 '$BASHPID' 는 다르게 나온다.

$ { echo '$BASHPID' : $BASHPID >&2 ;} > >( echo '$BASHPID' : $BASHPID )
$BASHPID : 504
$BASHPID : 22037


---------------------------------------------------------------------------

$ ls -l <( : ) 
lr-x------ 1 mug896 mug896 64 02.07.2015 22:29 /dev/fd/63 -> pipe:[681827]

$ [ -f <( : ) ]; echo $?  # 일반 파일인지 테스트
1
$ [ -p <( : ) ]; echo $?  # pipe 인지 테스트
0
```

임시 파일을 만들지 않고 ulimit의 내용을 비교 해보자.

```bash
$ ulimit -Sa > ulimit.Sa.out

$ ulimit -Ha > ulimit.Ha.out

$ diff ulimit.Sa.out ulimit.Ha.out
```

```bash
# 임시파일을 만들 필요가 없다

$ diff <( ulimit -Sa ) <( ulimit -Ha )   
1c1
< core file size          (blocks, -c) 0
---
> core file size          (blocks, -c) unlimited
8c8
< open files                      (-n) 1024
---
> open files                      (-n) 65536
12c12
< stack size              (kbytes, -s) 8192
---
> stack size              (kbytes, -s) unlimited
```

```bash
$ echo hello > >( wc )
      1       1       6

$ wc < <( echo hello )
      1       1       6

------------------------

# 입력과 출력용 프로세스 치환을 동시에 사용
$ f1() {
    cat "$1" > "$2"
}

$ f1 <( echo 'hi there' ) >( tr a-z A-Z )
HI THERE

------------------------------------------------------

# --log-file 옵션 값으로 입력 프로세스 치환이 사용됨
$ rsync -avH --log-file=>(grep -Fv .tmp > log.txt) src/ host::dst/

-----------------------------------------------------------------

# tee 명령을 이용해 결과를 4개의 입력 프로세스 치환으로 전달하여 처리
$ ps -ef | tee >(grep tom > toms-procs.txt) \
               >(grep root > roots-procs.txt) \
               >(grep -v httpd > not-apache-procs.txt) \
               >(sed 1d | awk '{print $2}' > pids-only.txt)

-----------------------------------------------------------------

# dd 명령에서 입력 파일로 사용
dd if=<( cat /dev/urandom | tr -dc A-Z ) of=outfile bs=4096 count=1
```

```bash
i=0
sort list.txt | while read -r line; do
  (( i++ ))
  ...
done

echo "$i lines processed"  

# 파이프로 인해 parent 변수 i 에 값을 설정할수 없어 항상 0 이 표시된다.
0 lines processed

------------------------------------
i=0
while read -r line; do
  (( i++ ))
  ...
done < <(sort list.txt)

echo "$i lines processed"   

# 프로세스 치환을 이용해 while 문이 현재 shell 에서 실행되어 i 값을 설정할수 있다.
12 lines processed
```

명령 실행 결과중 stderr만 전달하고 싶을때

```bash
$ command1 2> >( command2 ... )

# 파이프를 이용할 경우
$ command1 2>&1 > /dev/null | command2 ...
```

process substitution은 background로 실행된다.
parent process가 child process를 기다리는 방법을
소개한다.

```bash
#!/bin/bash

sync1=`mktemp`
sync2=`mktemp`

# 여기서 subshell ( ) 을 사용한것은  >( while read -r line ... ) 명령이 child process 가 되어 
# 종료되지 않고 계속해서 read 대기상태가 되는 것을 방지하기 위해서입니다.
( while read -r line; do
    case $line in
        aaa* ) echo "$line" >& $fd1 ;;
        bbb* ) echo "$line" >& $fd2 ;;
    esac
done ) \
    {fd1}> >( while read -r line; do echo "$line" | sed -e 's/x/y/g'; sleep 1; done; \
            rm "$sync1" ) \
    {fd2}> >( while read -r line; do echo "$line" | sed -e 's/x/z/g'; sleep 2; done; \
            rm "$sync2" ) \
    < <( for ((i=0; i<4; i++)); do echo aaaxxx; echo bbbxxx; done; echo ooops );

echo --- end 1 ---
while [ -e "$sync1" -o -e "$sync2" ]; do sleep 1; done
echo --- end 2 ---
```

## word splitting

IFS(internal field separater)에 저장된 값을 기준으로 단어들을
분리한다. 단어들을 분리한다는 것은 command line의 IFS값들을
space로 변환한다는 것을 의미한다.

```bash
$ AA="11X22X33Y44Y55"

$ echo $AA
11X22X33Y44Y55

$ IFS=XY

$ echo $AA
11 22 33 44 55

# 변수를 quote 하게 되면 단어 분리가 발생하지 않습니다.
$ echo "$AA"
11X22X33Y44Y55

$ ( IFS=:; for v in $PATH; do echo "$v"; done )
/usr/local/sbin
/usr/local/bin
/usr/sbin
/usr/bin
/sbin
/bin
. . . . 
. . . .
```

IFS의 기본값은 space, tab, newline이다. IFS가 unset일때 역시 기본값과
마찬가지다. IFS가 null이면 word splitting은 없다.

```bash
$ echo -n "$IFS" | od -a
0000000  sp  ht  nl

# space, tab, newline 이 모두 space 로 변경되어 표시됩니다.
$ echo $( echo -e "11 22\t33\n44" )
11 22 33 44

# quote 을 하면 단어 분리가 발생하지 않습니다.
$ echo "$( echo -e "11 22\t33\n44" )"
11 22   33
44
```

IFS설정

```bash
# bash 의 경우
bash$ IFS=$'\n'              # newline 으로 설정
bash$ IFS=$' \t\n'           # 기본값 으로 설정

# sh 의 경우
sh$ IFS='                    # newline 설정
'
sh$ IFS=$(echo " \n\t")      # 기본값 설정
                             # 여기서 \t 을 \n 뒤로 둔것은 $( ) 을 이용해 변수에 값을 대입할때는
                             # 마지막 newline 들이 제거되기 때문입니다.
                             # 첫번째 문자는 "$*" 값을 표시할때 구분자로 사용되므로 
                             # 위치가 바뀌면 안되겠습니다.

# 다음과 같이 할 수도 있습니다.
# IFS 값을 변경하기 전에 백업하고 복구하기
sh$ oIFS=$IFS                # 기존 IFS 값 백업
sh$ IFS='                    # IFS 을 newline 으로 설정하여 사용
'
...
...
sh$ IFS=$oIFS                # 기존 IFS 값 복구
```

word splitting 이 발생하는 예

```bash
$ dirname="쉘 스크립트 강좌"

# $dirname 변수에서 단어분리가 일어나 마지막 단어인 '강좌' 가 디렉토리 명이 되었다.
$ cp *.txt $dirname                     
cp: target '강좌' is not a directory

# $dirname 변수를 quote 하여 정상적으로 실행됨.
$ cp *.txt "$dirname"
OK

--------------------------------------------------------------

$ AA="one two three"

# $AA 하나의 변수값이지만 단어분리에 의해 3개의 인수가 되었다.
$ args.sh $AA
$1 : one
$2 : two
$3 : three

# $AA 하나의 변수값이지만 단어분리에 의해 ARR 원소 개수가 3개가 되었다.
$ ARR=( $AA )      
$ echo ${#ARR[@]}
3

# $AA 하나의 변수값이지만 단어분리에 의해 3개의 값이 출력되었다.
$ for num in $AA; do 
>    echo "$num"
>done
one
two
three
```

quote는 word splitting이 발생 하지 않게 한다.

```bash
AA="echo hello world"

# 단어분리가 일어나 echo 는 명령 hello world 는 인수가 됩니다.
$ $AA                 
hello world

# quote 을 하면 단어분리가 일어나지 않으므로 'echo hello world' 전체가 하나의 명령이 됩니다.
$ "$AA"               
echo hello world: command not found
```

word splitting은 variable expansion, command substitution과 함께
발생하기 때문에 옳바르게 word splitting이 되지 않는 경우가 있다.

```bash
$ set -f; IFS=:
$ ARR=(Arch Linux:Ubuntu Linux:Suse Linux:Fedora Linux)
$ set +f; IFS=$' \t\n'

$ echo ${#ARR[@]}    # 올바르게 분리 되지 않는다.   
5
$ echo ${ARR[1]}    
Linux:Ubuntu
```

`$AA` 은 word spltting이 옳바르게 된다.

```bash
$ AA="Arch Linux:Ubuntu Linux:Suse Linux:Fedora Linux"

$ set -f; IFS=:               
$ ARR=( $AA )
$ set +f; IFS=$' \t\n'

$ echo ${#ARR[@]}    # 올바르게 분리 되었다.
4
$ echo ${ARR[1]}       
Ubuntu Linux
```

IFS값을 `Q`로 바꾸었지만 space를 기준으로 wordsplitting이 발생

```bash
f1() {
    echo \$1 : "$1"
    echo \$2 : "$2"
}

IFS=Q          # IFS 값을 'Q' 로 설정

f1 11Q22
f1 33 44
====== output ========

$1 : 11Q22     # 'Q' 에 의해 인수가 분리되지 않는다.
$2 :
$1 : 33        # 그대로 공백에 의해 인수가 분리된다.
$2 : 44
```

IFS값을 `Q`로 바꾸고 variable expansion이 발생하면 
`Q`를 기준으로 wordsplitting이 발생.

```bash
IFS=Q     
AA="11Q22"
BB="33 44"
f1 $AA
f1 $BB
====== output ========

$1 : 11        # 'Q' 에 의해 인수가 분리된다.
$2 : 22
$1 : 33 44     # 공백 에서는 분리되지 않는다.
$2 :
```

IFS값이 white space (space, tab, newline)일 경우와 아닐 경우의 차이

```bash
$ AA="11          22"
$ IFS=' '              # IFS 값이 공백문자일 경우
$ echo $AA 
11 22                  # 연이어진 공백문자는 하나로 줄어든다.

# IFS 값이 기본값일 경우
$ echo $( echo -e "11       22\t\t\t\t33\n\n\n\n44" )
11 22 33 44

$ AA="11::::::::::22"
$ IFS=':'              # IFS 값이 공백문자가 아닐 경우
$ echo $AA         
11          22         # 줄어들지 않고 모두 공백으로 표시된다.
---------------------------

$ AA="Arch:Ubuntu:::Mint"

$ IFS=:                 # 공백이 아닌 문자를 사용하는 경우

$ ARR=( $AA )

$ echo ${#ARR[@]}       # 원소 개수가 빈 항목을 포함하여 5 개로 나온다.
5

$ echo ${ARR[1]}
Ubuntu
$ echo ${ARR[2]}

$ echo ${ARR[3]}

-----------------------------------

AA="Arch Ubuntu        Mint"

$ IFS=' '               # 공백문자를 사용하는 경우

$ ARR=( $AA )

$ echo ${#ARR[@]}       # IFS 값이 공백문자일 경우 연이어진 공백문자들은 하나로 취급됩니다.
3                       # 그러므로 원소 개수가 3 개로 나온다

$ echo ${ARR[1]}
Ubuntu
$ echo ${ARR[2]}
Mint
```

파일 이름에 space가 포함되어 있는 경우 word splitting때문에 파일
이름이 분리 될 수 있다. IFS값을 newline으로 변경하면 해결 가능하다.

```bash
$ ls
2013-03-19 154412.csv  ReadObject.java       WriteObject.class
ReadObject.class       쉘 스크립트 테스팅.txt    WriteObject.java


$ for file in $(find .)
do
        echo "$file"
done
.
./WriteObject.java
./WriteObject.class
./ReadObject.java
./2013-03-19            # 파일이름이 2개로 분리
154412.csv
./ReadObject.class
./쉘                    # 파일이름이 3개로 분리
스크립트
테스팅.txt

-----------------------------------------------

$ set -f; IFS=$'\n'           # IFS 값을 newline 으로 변경
                              # set -f 는 globbing 방지를 위한 옵션 설정
$ for file in $(find .)
do
        echo "$file"
done
.
./WriteObject.java
./WriteObject.class
./ReadObject.java
./2013-03-19 154412.csv
./ReadObject.class
./쉘 스크립트 테스팅.txt

$ set +f; IFS=$' \t\n'
```

## filename expansion

파일 이름을 다룰때 `*`, `?`, `[]`는 패턴 매칭과 동일하게 확장된다.
앞서 언급한 문자들을 glob 문자라고 한다. glob 문자가 확장되는 것을
globbing이라고 한다.

```bash
$ ls
address.c      address.h     readObject.c      readObject.h     WriteObject.class
Address.class  Address.java  ReadObject.class  ReadObject.java  WriteObject.java

$ ls *.[ch]
address.c  address.h  readObject.c  readObject.h

$ ls "*.[ch]"         # quote 을 하면 globbing 이 일어나지 않는다
ls: cannot access *.[ch]: No such file or directory

$ echo *.?
address.c address.h readObject.c readObject.h

$ for file in *.[ch]; do
      echo "$file"
done

address.c
address.h
readObject.c
readObject.h
```

bash전용 옵션인 globstar를 이용하여 recursive matching을 수행

```bash
# globstar 옵션은 기본적으로 off 상태 이므로 on 으로 설정해 줍니다.
$ shopt -s globstar

# 현재 디렉토리 이하 에서 모든 디렉토리, 파일 select
$ for dirfile in **; do
      echo "$dirfile"
done

# 현재 디렉토리 이하 에서 모든 디렉토리 select
$ for dir in **/; do
      echo "$dir"
done

# 현재 디렉토리 이하 에서 확장자가 java 인 파일 select
$ for javafile in **/*.java; do
      echo "$javafile"
done

# $HOME/tmp 디렉토리 이하 에서 확장가 java 인 파일 select
$ for javafile in ~/tmp/**/*.java; do
      echo "$javafile"
done

# $HOME/tmp 디렉토리 이하 에서 디렉토리 이름에 log 가 포함될 경우 select
$ for dir in ~/tmp/**/*log*/; do
      echo "$dir"
done

# brace 확장과 함께 사용
$ for imgfile in ~/tmp/**/*.{png,jpg,gif}; do
      echo "$imgfile"
done


$ awk 'BEGINFILE { print FILENAME; nextfile }' **/*.txt
....
....
```

globbing으로 부족하여 find를 이용하는 경우

```bash
for file in $(find . -type f) ...  # Wrong!
for file in `find . -type f` ...   # Wrong!

arr=( $(find . -type f) ) ...      # Wrong!

-------------------------------------------
# 파일명에 공백이나 glob 문자가 사용되지 않는다면 위와 같이 해도 됩니다.
# 하지만 그렇지 않을 경우 단어분리, globbing 에대한 처리를 해주어야 합니다.

set -f; IFS=$'\n'
for file in $(find . -type f) ...  
set +f; IFS=$' \t\n'

# subshell 을 이용
( set -f; IFS=$'\n'
for file in $(find . -type f) ...  )

set -f; IFS=$'\n'
arr=( $(find . -type f) ) ...  
set +f; IFS=$' \t\n'
```

find는 파일 이름을 출력할때 newline을 구분자로 이용한다. 그러나
`-print0`옵션을 사용하면 `NUL`을 구분자로 이용하여 출력 할 수 있다.

```bash
# find 명령에서 -print0 을 이용해 출력했으므로 read 명령의 -d 옵션 값을 null 로 설정
find . -type f -print0 | while read -r -d '' file
do
    echo "$file"
done

# 프로세스 치환을 이용
while read -r -d '' file; do
    echo "$file"
done < <( find . -type f -print0 )

# 명령치환은 NUL 문자를 전달하지 못하므로 -print0 옵션과 함께 사용할 수 없습니다.
```

bash와 달리 sh는 globstar옵션을 사용 할 수 없다. `read -d` 역시 마찬가지 이다.
그러나 find와 for를 이용하여 word splitting, globbing, newline을
문제 없이 처리 할 수 있다.

```sh
# 다음과 같이 실행할 경우 select 된 파일이 3개라면 각각 3 번의 sh -c 명령이 실행되는 것과 같습니다.
sh$ find -type f -exec sh -c 'mv "$0" "${0%.*}.py"' {} \;
# 1. sh -c '...' "file1"
# 2. sh -c '...' "file2"
# 3. sh -c '...' "file3"

# for 문에서 in words 부분을 생략하면 in "$@" 와 같게 되고 "$1" "$2" "$3" ... 차례로 입력됩니다. 
# sh -c 'command ...' 11 22 33 형식으로 명령을 실행하면 command 의 첫번째 인수에 해당하는 11 이 
# "$0" 에 할당되어 for 문에서 사용할 수 없게 되므로 임의의 문자 X 를 사용하여 
# select 된 파일명이 "$1" 자리에 오게 하였습니다.
# 좀 더 효율적으로 명령을 실행하기 위해 find 명령의 마지막에 + 기호를 사용함으로써
# 한번의 sh -c 명령으로 실행을 완료할 수 있습니다.
sh$ find -type f -exec \
    sh -c 'for file; do mv "$file" "${file%.*}.py"; done' X {} +
# 1. sh -c '...'  X "file1" "file2" "file3"
```

xargs를 활용하는 방법

```sh
# find 명령에서 -print0 을 이용하여 출력하였으므로 xargs 에서 -0 옵션을 사용하였습니다.
# xargs 명령에서 사용된 {} 와 X 의 의미는 위 예제와 같고 {} 는 select 된 파일명을 나타냅니다.
sh$ find -type f -print0 | 
    xargs -0i sh -c 'mv "$0" "${0%.*}".py' {}
# 1. sh -c '...' "file1"
# 2. sh -c '...' "file2"
# 3. sh -c '...' "file3"

sh$ find -type f -print0 | 
    xargs -0 sh -c 'for file; do mv "$file" "${file%.*}".py; done' X
# 1. sh -c '...'  X "file1" "file2" "file3"
```

unset의 인수로 사용된 array[12]가 globbing에 의해 파일 array1
과 매칭이 되어 unset이 되지 않고 있다.

```bash
$ array=( [10]=100 [11]=200 [12]=300 )
$ echo ${array[12]}
300

$ touch array1               # 현재 디렉토리에 임의로 array1 파일생성

# unset 을 실행하였으나 globbing 에의해 array[12] 이 array1 파일과 매칭이되어 
# 값이 그대로 남아있음
$ unset -v array[12]         
$ echo ${array[12]}         
300                       

$ unset -v 'array[12]'       # globbing 을 disable 하기위해 quote.
$ echo ${array[12]}          # 이제 정상적으로 unset 이됨

-----------------------------------------

# tr 명령을 이용해 non printable 문자를 삭제하려고 하지만
# [:graph:] 가 파일 a 와 매칭이 되어 실질적으로 'tr -dc a' 와 같은 명령이 되었습니다.

$ touch a

$ head -c100 /dev/urandom | tr -dc [:graph:]

# 다음과 같이 quote 합니다.
$ head -c100 /dev/urandom | tr -dc '[:graph:]'
```

array에 입력되는 원소 값에 glob문자가 포함됨

```bash
$ AA="Arch Linux:*:Suse Linux:Fedora Linux"

$ IFS=:
$ ARR=($AA)
$ IFS=$' \t\n'

$ echo "${ARR[@]}"
Arch Linux 2013-03-19 154412.csv Address.java address.ser 
ReadObject.class ReadObject.java 쉘 스크립트 테스팅.txt 
WriteObject.class WriteObject.java Suse Linux Fedora Linux

-----------------------------------------------------------

$ set -f; IFS=:          # globbing 을 disable
$ ARR=($AA)
$ set +f; IFS=$' \t\n'

$ echo "${ARR[@]}"
Arch Linux * Suse Linux Fedora Linux
```

현재 디렉토리 이하에서 확장자가 .c인 파일을 모두 찾으려고 하지만
glob문자에 의해 aaa.c파일과 매칭이 되어서 `find -name aaa.c`
와 같이 되었다.

```bash
$ touch aaa.c

$ find -name *.c
aaa.c

# 다음과 같이 quote 합니다.
$ find -name '*.c'
./aaa.c
./fork/fork.c
./write/write.c
./read/read.c
...
```

## quote removal

지금까지 expansion에 포함되지 않고 quote되지 않은 `\`, `'`, `"` 캐릭터는 제거된다.

```bash
echo "hello" \ \
```

# Shell Builtin Commands

## Bourne Shell Builtins

`:`, `.`, `break`, `cd`, `continue`, `eval`, `exec`, `exit`, `export`, `getopts`, `hash`, `pwd`, `readonly`, `return`, `shift`, `shift`, `test`, `times`, `trap`, `umask`, `unset`

## Bash Builtins

`alias`, `bind`, `builtin`, `caller`, `command`, `declare`, `echo`, `enable`, `help`, `let`, `local`, `logout`, `mapfile`, `printf`, `read`, `readarray`, `source`, `type`, `typeset`, `ulimit`, `unalias`

# Shell Variables

## Bourne Shell Variables

* `CDPATH`
  * 현재 작업디렉토리
  * `echo $CDPATH` 
* `HOME`
  * 사용자의 집디렉토리
  * `echo $HOME`
* `IFS`
  * 파라미터 구분자
  * `echo $IFS`
* `MAIL`
* `MAILPATH`
* `OPTARG`
* `OPTIND`
* `PATH`
* `PS1`
* `PS2`

## Bash Variables

* `BASH`
* `BASHOPTS`

# Bash Features

## Bash Startup Files

ssh, telnet등으로 접속하여 login한 다음 얻은 shell을 login shell이라고 한다. 윈도우 매니저에서 메뉴를 통해 얻은 shell을 non-login shell이라고 한다.

### login shell

시작시 `/etc/profile`, `~/.bash_profile`, `~/.bash_login`, `~/.profile`을 실행한다. 보통 `/etc/profile` 에서 `source /etc/bash.bashrc` 하고 `~/.profile` 에서 `source ~/.bashrc` 한다.

로그아웃시 `~/.bash_logout`을 실행한다.

`shopt -s huponexit` 옵션 설정을 통해 로그아웃시 background job들에게 HUP 시그널을 보낸다.

### non-login shell

시작시 `/etc/bash.bashrc`, `~/.bashrc` 를 실행한다. `rc`는 `run commands`의 약어이다. [참고](http://www.catb.org/jargon/html/R/rc-file.html)

`su - userid` 혹은 `bash -l` 을 이용하여 login shell을 만들 수 있다.

## Bash Conditional Expressions

conditional expressions는 `[[` compound command 그리고 `test`, `[` builtin command 에 의해 사용된다.

```
-a file
True if file exists.

-b file
True if file exists and is a block special file.

-c file
True if file exists and is a character special file.

-d file
True if file exists and is a directory.

-e file
True if file exists.

-f file
True if file exists and is a regular file.

-g file
True if file exists and its set-group-id bit is set.

-h file
True if file exists and is a symbolic link.

-k file
True if file exists and its "sticky" bit is set.

-p file
True if file exists and is a named pipe (FIFO).

-r file
True if file exists and is readable.

-s file
True if file exists and has a size greater than zero.

-t fd
True if file descriptor fd is open and refers to a terminal.

-u file
True if file exists and its set-user-id bit is set.

-w file
True if file exists and is writable.

-x file
True if file exists and is executable.

-G file
True if file exists and is owned by the effective group id.

-L file
True if file exists and is a symbolic link.

-N file
True if file exists and has been modified since it was last read.

-O file
True if file exists and is owned by the effective user id.

-S file
True if file exists and is a socket.

file1 -ef file2
True if file1 and file2 refer to the same device and inode numbers.

file1 -nt file2
True if file1 is newer (according to modification date) than file2, or if file1 exists and file2 does not.

file1 -ot file2
True if file1 is older than file2, or if file2 exists and file1 does not.

-o optname
True if the shell option optname is enabled. The list of options appears in the description of the -o option to the set builtin (see The Set Builtin).

-v varname
True if the shell variable varname is set (has been assigned a value).

-R varname
True if the shell variable varname is set and is a name reference.

-z string
True if the length of string is zero.

-n string
string
True if the length of string is non-zero.

string1 == string2
string1 = string2
True if the strings are equal. When used with the [[ command, this performs pattern matching as described above (see Conditional Constructs).

‘=’ should be used with the test command for POSIX conformance.

string1 != string2
True if the strings are not equal.

string1 < string2
True if string1 sorts before string2 lexicographically.

string1 > string2
True if string1 sorts after string2 lexicographically.

arg1 OP arg2
OP is one of ‘-eq’, ‘-ne’, ‘-lt’, ‘-le’, ‘-gt’, or ‘-ge’. These arithmetic binary operators return true if arg1 is equal to, not equal to, less than, less than or equal to, greater than, or greater than or equal to arg2, respectively. Arg1 and arg2 may be positive or negative integers.
```

## Shell Arithmetic

shell arithmetic expressions 는 `((` compound command, `let` builtin command 혹은 `-i` 옵션의 `declare` builtin command 에서 사용된다.

```
id++ id--
variable post-increment and post-decrement

++id --id
variable pre-increment and pre-decrement

- +
unary minus and plus

! ~
logical and bitwise negation

**
exponentiation

* / %
multiplication, division, remainder

+ -
addition, subtraction

<< >>
left and right bitwise shifts

<= >= < >
comparison

== !=
equality and inequality

&
bitwise AND

^
bitwise exclusive OR

|
bitwise OR

&&
logical AND

||
logical OR

expr ? expr : expr
conditional operator

= *= /= %= += -= <<= >>= &= ^= |=
assignment

expr1 , expr2
comma
```

## Arrays

array를 만들어 보자.

```bash
A=( 11 "hello world" 22 )

A=( [0]=11 [1]="hello world" [2]=22)

A[0]=11
A[1]="hello world"
A[2]=22
```

associative array를 만들어 보자.

```bash
declare -A A
A=( [ab]=11 [cd]="hello world" [ef]=22 )

A[ab]=11
A[cd]="hello world"
A[ef]=22
```

array를 조회해 보자.

```bash
A=(foo bar baz)
declare -p A
```

현재 shell env에 정의된 array 보기

```bash
compgen -A arrayvar
```

array를 사용할때 paramater expansion을 이용해야 한다.

```bash
AA=(11 22 33)
# '112' 라는 파일이 있다면 globbing이 발생한다.
echo $AA[2]
# 반드시 다음과 같이 paramater expansion을 이용하자.
echo ${AA[2]}
```

## Controlling the Prompt

다음은 prompt variable 에 해당하는 `PS1` 부터 `PS4` 에 등장하는 special characters이다.

```
\a
A bell character.

\d
The date, in "Weekday Month Date" format (e.g., "Tue May 26").

\D{format}
The format is passed to strftime(3) and the result is inserted into the prompt string; an empty format results in a locale-specific time representation. The braces are required.

\e
An escape character.

\h
The hostname, up to the first ‘.’.

\H
The hostname.

\j
The number of jobs currently managed by the shell.

\l
The basename of the shell’s terminal device name.

\n
A newline.

\r
A carriage return.

\s
The name of the shell, the basename of $0 (the portion following the final slash).

\t
The time, in 24-hour HH:MM:SS format.

\T
The time, in 12-hour HH:MM:SS format.

\@
The time, in 12-hour am/pm format.

\A
The time, in 24-hour HH:MM format.

\u
The username of the current user.

\v
The version of Bash (e.g., 2.00)

\V
The release of Bash, version + patchlevel (e.g., 2.00.0)

\w
The current working directory, with $HOME abbreviated with a tilde (uses the $PROMPT_DIRTRIM variable).

\W
The basename of $PWD, with $HOME abbreviated with a tilde.

\!
The history number of this command.

\#
The command number of this command.

\$
If the effective uid is 0, #, otherwise $.

\nnn
The character whose ASCII code is the octal value nnn.

\\
A backslash.

\[
Begin a sequence of non-printing characters. This could be used to embed a terminal control sequence into the prompt.

\]
End a sequence of non-printing characters.
```

# Job Control

## Job Control Builtins

```bash
bg, fg, jobs, kill, wait, disown, suspend
```
