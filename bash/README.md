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
- [Shell Commands](#shell-commands)
  - [Shell Builtin Commands](#shell-builtin-commands)
    - [Bourne Shell Builtins](#bourne-shell-builtins)
    - [Bash Builtins](#bash-builtins)
  - [Shell Functions](#shell-functions)
  - [Shell Keywords](#shell-keywords)
  - [Pipelines](#pipelines)
  - [List of Commands](#list-of-commands)
  - [Compound Commands](#compound-commands)
    - [Looping Constructs](#looping-constructs)
    - [Conditional Constructs](#conditional-constructs)
    - [Grouping Commands](#grouping-commands)
  - [Tips](#tips)
- [Shell Options](#shell-options)
  - [set](#set)
  - [shopt](#shopt)
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
- [Test](#test)
- [Special Expressions](#special-expressions)
- [Sub Shell](#sub-shell)
- [Job Control](#job-control)
  - [Job Control Builtins](#job-control-builtins)
- [Signals](#signals)
  - [signal list](#signal-list)
  - [kill](#kill)
  - [trap](#trap)
- [Problems](#problems)

-------------------------------------------------------------------------------

# Abstract

bash 에 대해 정리한다.

# References

* [Bash script](https://www.gitbook.com/book/mug896/shell-script/details)
  * 한글 가이드
* [bash reference manual @ gnu](https://www.gnu.org/software/bash/manual/bash.html)
* [Advanced Bash-Scripting Guide](http://www.tldp.org/LDP/abs/html/index.html)
* [bash repo](https://savannah.gnu.org/git/?group=bash)

# Shell Metachars

metachars 는 command 와 다르게 처리되기 때문에 command line 에 포함하는 경우 escape 하거나 quote 해서 사용해야 한다.

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
# help command
help echo

# type command
$ type ssh
ssh is /usr/bin/ssh
$ type caller
caller is a shell builtin
$ type time
time is a shell keyword

# list keywords
$ compgen -k | column
if              elif            esac            while           done            time            !               coproc
then            fi              for             until           in              {               [[
else            case            select          do              function        }               ]]

# list builtins
$ compgen -b | column
.               caller          dirs            false           kill            pwd             source          ulimit
:               cd              disown          fc              let             read            suspend         umask
[               command         echo            fg              local           readarray       test            unalias
alias           compgen         enable          getopts         logout          readonly        times           unset
bg              complete        eval            hash            mapfile         return          trap            wait
bind            compopt         exec            help            popd            set             true
break           continue        exit            history         printf          shift           type
builtin         declare         export          jobs            pushd           shopt           typeset

# file name can be anything except NUL, / on linux
echo "hello world" > [[]].txt

# '-' can be stdin for input
echo hello world | cat -
# '-' can be stdout for output
echo hello world | cat a.txt -
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
echo "foo" &> a.txt

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

# brace expansion
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
while 1
do
    echo "foo";
    sleep 1;
done
while 1; do echo "foo"; sleep 1; done

# for
for i in $(l)
do 
    echo $i
done
for i in ${1..10}; do echo ${i}; done

for (( i=0; i < 10; i++ ))
do
    echo $i
done
for (( i=0; i<10; i++ )); do echo $i; done

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
# (command line) : command line 은 subshell 환경에서 실행된다.
( while true; do echo "hello"; sleep 1; done )
# {command line} : command line 은 같은 shell 환경에서 실행된다.
{ while true; do echo "hello"; sleep 1; done }

# variable 은 unset, null, not-null 과 같이 3 가지 상태를 갖는다.
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

# function 은 command 들을 그룹화 하는 방법이다. 그룹의 이름을 사용하면 그룹안의 commands를 실행할 수 있다.
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

# shell file의 첫줄에서 스크립트를 실행할 command line 을 shabang line 이라고 한다. 옵션은 하나로 제한된다.
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

shell 이 command 를 읽고 실행하는 과정은 다음과 같다.

* commands 및 arguments 를 읽어 들인다.
* 읽어 들인 commands 를 quoting 과 동시에 words 와 operators 로 쪼갠다. 쪼개진 토큰들은 metacharaters 를 구분자로 한다. 이때 alias expansion 이 수행된다.
* 토큰들을 읽어서 simple, compound commands 를 구성한다.
* 다양한 shell expansion 이 수행되고 expanded tokens 는 파일이름, 커맨드, 인자로 구성된다.
* redirection 을 수행하고 redirection operators 와 operands 를 argument list 에서 제거한다.
* command 를 실행한다.
* 필요한 경우 command 가 실행 완료될 때까지 기다리고 exit status 를 수집한다.

# Shell Parameters

## Positional Parameters

`${N}` 처럼 표기한다. N 는 paramter 의 서수이다.

* a.sh `echo ${0}, ${1}, ${2}` 
  * `echo a.sh "hello" "world" "foo"`

* `shift`
  * positional parameters 를 좌측으로 n 만큼 이동한다.

```bash
$ set -- 11 22 33 44 55
$ echo $@
11 22 33 44 55
$ shift 2
$ echo $@
33 44 55
```

## Special Parameters

* `$*`, `$@`
  * 모든 parameter 들
  * a.sh `echo $*` 
    * `bash a.sh 1 2 3 4 5`
      * 결과는 `1 2 3 4 5`

* `$#`
  * 마지막 parameter
  * a.sh `echo $#` 
    * `bash a.sh 1 2 3 4 5`
      * 결과는 `5`

* `$?`
  * 마지막 실행된 foreground pipeline 의 exit status
  * a.sh `echo $?` 
    * `bash a.sh 1 2 3 4 5`
      * 결과는 `0`

* `$-`
  * 현재 설정된 option flags
  * a.sh `echo $-` 
    * `bash a.sh 1 2 3 4 5` 
      * 결과는 `hB`
  
* `$$`
  * 실행 프로세스 ID
  * a.sh `echo $$` 
    * `bash a.sh 1 2 3 4 5`
      * 결과는 `3`

* `$!`
  * 가장 최근에 실행된 background 프로세스 ID
  * a.sh `echo $!` 
    * `bash a.sh 1 2 3 4 5`
      * 결과는 ``

* `$0`
  * shell or shell script
  * a.sh `echo $!` 
    * `bash a.sh 1 2 3 4 5`
      * 결과는 `a.sh`

* `$_`
  * shell 혹은 shell script 의 절대경로
    * a.sh `echo $_` `bash a.sh 1 2 3 4 5`
      * 결과는 `/bin/bash`

# Shell Expansions

command line 은 tokens 로 나눠진 후 다음과 같은 순서 대로 expansion 처리
된다. brace expansion, tilde expansion, parameter and variable
expansion, command expansion, arithmetic expansion, process
substitution, word splitting, filename expansion, quote removal

## brace expansion

### string lists

보통 다음과 같은 형식으로 사용한다.

```
{string1, string2,..., stringN}
```

`,` 가 사용되지 않은 단일 항목은 확장되지 않는다. `,` 전 후에 공백을
사용할 수는 없다. 문자열에 공백이 포함되는 경우 quote 해야
expansion 할 수 있다.

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

preamble 과 postscript 의 사용

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

null 값과 사용

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

preamble 혹은 postscript 의 사용

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

```bash
$ echo ~
/Users/iamslash
$ echo ~iamslash # iamslash 유저 디렉토리를 출력하라.
/Users/iamslash
$ echo ~+ # $PWD 와 같다.
$ echo ~- # $OLDPWD 와 같다.
```

## parameter and variable expansion

### basic usage

`{}` 를 사용한다.

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

command substitution 은 subshell에서 실행된다. 실행결과에 해당하는 stdout
 값이 pipe 를 통해 전달된다. 일종의 IPC 이다.

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

backtick 은 escape sequence 가 다르게 처리된다.

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
(( arithmetic-expr ))
```

산술연산을 계산한다.

```bash
$ cat > a.sh
echo $(( 3 + 7 ))
$ bash a.sh
10
```

## process substitution

```bash
<( <COMMANDS> )
>( <COMMANDS> )
```

`command1 > >( command2 )` 명령의 경우 command1 의 stdout 이 command2 의 stdin 과 연결되며 `command1 < <( command2 )` 명령의 경우는 command2 의 stdout 이 command1 의 stdin 과 연결됩니다.  현재 shell pid 를 나타내는 $$ 변수는 subshell 에서도 동일한 값을 가지므로 >( ) 표현식 내에서의 FD 상태를 보기 위해서는 $BASHPID 변수를 이용해야 합니다.

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

명령 실행 결과중 stderr 만 전달하고 싶을때

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

지금까지 expansion 에 포함되지 않고 quote 되지 않은 `\`, `'`, `"` 캐릭터는 제거된다.

```bash
echo "hello" \ \
```

# Shell Commands

## Shell Builtin Commands

```bash
$ compgen -b | column
.               compopt         fg              pushd           trap
:               continue        getopts         pwd             true
[               declare         hash            read            type
alias           dirs            help            readarray       typeset
bg              disown          history         readonly        ulimit
bind            echo            jobs            return          umask
break           enable          kill            set             unalias
builtin         eval            let             shift           unset
caller          exec            local           shopt           wait
cd              exit            logout          source
command         export          mapfile         suspend
compgen         false           popd            test
complete        fc              printf          times
```

### Bourne Shell Builtins

`:`, `.`, `break`, `cd`, `continue`, `eval`, `exec`, `exit`, `export`, `getopts`, `hash`, `pwd`, `readonly`, `return`, `shift`, `shift`, `test`, `times`, `trap`, `umask`, `unset`

### Bash Builtins

`alias`, `bind`, `builtin`, `caller`, `command`, `declare`, `echo`, `enable`, `help`, `let`, `local`, `logout`, `mapfile`, `printf`, `read`, `readarray`, `source`, `type`, `typeset`, `ulimit`, `unalias`

## Shell Functions

```bash
$ declare -F
```

## Shell Keywords

```bash
$ compgen -k | column
if              case            until           time            ]]
then            esac            do              {               coproc
else            for             done            }
elif            select          in              !
fi              while           function        [[
```

## Pipelines

```
$ echo hello world | cat -
```

## List of Commands

```
$ echo hello; echo world;
$ make && make install
$ echo "hello" || echo "world"
```

## Compound Commands

### Looping Constructs

```bash
until test-commands; do consequent-commands; done
while test-commands; do consequent-commands; done
for name [ [in [words …] ] ; ] do commands; done
for (( expr1 ; expr2 ; expr3 )) ; do commands ; done

$ read -p "Enter Hostname: " hostname
$ until ping -c 1 "$hostname" > /dev/null; do sleep 60; done; curl -O "$hostname"

$ while read -r line < a.text; do echo "$line"; done

$ set -f; IFS=$'\n'
$ for file in $(find -type f); do echo "$file"; done;

$ for (( i = 0; i <= 5; i++ )) { echo $i; }
```
### Conditional Constructs

```bash
if, case, select, ((...)), [[...]]
```

```bash
$ if grep -q 'hello world' a.txt; then echo "found it"; fi

read -p "Enter the name of an animal: " ANIMAL
echo -n "The $ANIMAL has "
case $ANIMAL in
    horse | dog | cat) 
        echo -n 4
        ;;
    man | kangaroo) 
        echo -n 2
        ;;
    *) 
        echo -n "an unknown number of"
        ;;
esac
echo " legs."

##### select
$ A=( "foo" "bar" "baz" )
$ PS3="Enter number: "
$ select KEY in "${A[@]}"; do echo "$KEY"; done

##### 
```

### Grouping Commands

`;`, `&`, `&&`, `||` 를 활용한 command

```bash
# (command list) : command list는 subshell환경에서 실행된다.
( while true; do echo "hello"; sleep 1; done )
# {command list} : command list는 같은 shell환경에서 실행된다.
{ while true; do echo "hello"; sleep 1; done }
```

## Tips

이름은 같고 의미가 다른 command 를 조사하고 싶을때는 `type` 을 이용하자.

```bash
$ type -a kill
kill is a shell builtin
kill is /bin/kill
$ type -a time
time is a shell keyword
time is /usr/bin/time
$ type -a [
[ is a shell builtin
[ is /usr/bin/[
```

command의 help를 보는 방법은 다음과 같다.

```bash
$ man -f printf
printf (1)           - format and print data
$ man 1 printf
# regex search
$ man -k print? 
$ info printf
```

다음은 man의 secion에 대한 내용이다.

| section |	desc |	example
|------|-------|------|  
| 1 |	User Commands	| |
| 2 |	System Calls	| man 2 write |
| 3 |	C Library Functions	man 3 printf | |
| 4 |	Devices and Special Files (usually found in /dev) |	man 4 tty |
| 5 |	File formats and conventions e.g /etc/passwd, /etc/crontab |	man 5 proc |
| 6 |	Games	| |
| 7 |	Miscellaneous (including macro packages and conventions)	| man 7 signal, man 7 hier |
| 8 |	System Administration tools and Deamons (usually only for root)	 | |

# Shell Options

shell의 옵션은 `set`과 `shopt`를 이용하여 설정할 수 있다. `shopt`는 bash 전용이다. 옵션의 설정값은 sh의 경우 `SHELLOPTS`에 bash의 경우 `BASHOPTS`에 저장된다.

## set

`--`는 현재 설정되어있는 positional parameters를 삭제한다.

```bash
$ set 11 22 33
$ echo "$@"
11 22 33
$ set --
$ echo "$@"
```

## shopt

설정한 옵션값은 `shopt` 혹은 `shopt -p` 으로 확인할 수 있다. 옵션값을 enable 할때는 `shopt -s 옵션이름` , disable 할때는 `shopt -u 옵션이름` 을 사용한다.

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

array를 사용할때 paramater expansion을 이용해야 한다. 그렇지 않으면 다음과 같이 side effect가 발생할 수 있다.

```bash
AA=(11 22 33)
# '112' 라는 파일이 있다면 globbing이 발생한다.
echo $AA[2]
# 반드시 다음과 같이 paramater expansion을 이용하자.
echo ${AA[2]}
```

`${AA[@]}`, `${AA[*]}` 는 double quote 하지 않으면 차이가 없다. `"${AA[@]}"` 는 `"${AA[0]}" "${AA[1]}" "${AA[2]}" ...`과 같다. `"${AA[*]}"` 는 `"${AA[0]}X${AA[1]}X${AA[2]}...`과 같다. `X`는 IFS의 첫번째 캐릭터이다.

```bash
# quote 하지 않은 경우
$ AA=( "hello   world" "foo   bar" baz )
$ echo ${#AA[@]}
3
$ echo ${AA[@]}
hello world foo bar baz
$ echo ${AA[*]}
hello world foo bar baz
$ for v in ${AA[@]}; do echo "$v"; done
hello 
world 
foo   
bar   
baz   
$ for v in ${AA[*]}; do echo "$v"; done
hello 
world 
foo   
bar   
baz   
# quote한 경우
$ echo "${AA[@]}"
$ echo "${AA[*]}"
# 배열의 원래 개수를 유지한다.
$ for v in "${AA[@]}"; do echo "$v"; done
hello   world 
foo   bar     
baz           
# 배열의 원소들이 하나로 합쳐진다.
$ for v in "${AA[*]}"; do echo "$v"; done
hello   world foo   bar baz
```

array의 특수표현을 살펴보자.

| expression | meaning |
|------------|---------|
| `${#array[@]}` `${#array[*]}` | array 전체 원소의 개수 |
| `${#array[N]}` `${#array[string]}` | indexed array 에서 N 번째 원소의 문자 수를 나타냄 associative array 에서 index 가 string 인 원소의 문자 수를 나타냄 |
| `${array[@]}` `${array[*]}` | array 전체 원소 |
| `${!array[@]}` `${!array[*]}` | array 전체 인덱스 |
| `${!name@}` `${!name*}` | name 으로 시작하는 이름을 갖는 모든 변수를 나타냄 |

array를 순회하자.

```bash
# indexed array
AA=(11 22 33)
for idx in ${!AA[@]}; do echo index : $idx, value : "${AA[idx]}";done
# associative array
declare -A AA
AA=( [ab]=11 [cd]="hello world" [ef]=22 )
for idx in "${!AA[@]}"; do echo index : "$idx", value : "${AA[idx]}"; done
```

array를 복사해보자. 항상 `()`를 사용하자.

```bash
$ AA=( 11 22 33 )
$ BB=${AA[@]}
$ echo "${BB[1]}" # 비정상
$ echo "$BB"
11 22 33
$ echo "${BB[0]}"
11 22 33
$ BB=( "${AA[@]}" )
$ echo "${BB[1]}"
22
```

array를 삭제해 보자.

| command | meaning |
|---------|---------|
| `array=()` `unset -v array` `unset -v "array[@]"` | array 삭제 |
| `unset -v "array[N]"` | indexed array에서 N번째 원소 삭제 |
| `unset -v "array[string]"` | associative array에서 index가 string인 원소 삭제 |

```bash
$ AA=(11 22 33 44 55)
$ unset -v "AA[2]"
$ echo "${AA[@]}"
11 22 44 55
$ for v in "${AA[@]}"; do echo "$v"; done  
11
22
44         
55
# AA[2]가 비어있다.
$ echo ${AA[1]} : ${AA[2]} : ${AA[3]}
22 : : 44
$ AA=( "${AA[@]}" )
$ echo ${AA[1]} : ${AA[2]} : ${AA[3]}
22 : 44 : 55
```

null 값을 가지고 있는 원소를 삭제해 보자.

```bash
$ AA=":arch linux:::ubuntu linux::::fedora linux::"
$ IFS=: read -ra ARR <<<  "$AA"
$ echo ${#ARR[@]}
10
$ echo ${ARR[0]}

$ echo ${ARR[1]}
arch linux
$ set -f; IFS=''
$ ARR=( ${ARR[@]} ) 
$ set +f; IFS=$' \n\t'
$ echo ${#ARR[@]}
3
$ echo ${ARR[0]}
arch linux
$ echo ${ARR[1]}
ubuntu linux
```

array의 원소를 ` ${array[@]:offset:length}`를 이용하여 추출할 수 있다.

```bash
$ AA=( Arch Ubuntu Fedora Suse Mint );

$ echo "${AA[@]:2}"
Fedora Suse Mint

$ echo "${AA[@]:0:2}"
Arch Ubuntu

$ echo "${AA[@]:1:3}"
Ubuntu Fedora Suse
```

array에 원소를 추가해보자.

```bash
$ AA=( "Arch Linux" Ubuntu Fedora);

$ AA=( "${AA[@]}" AIX HP-UX);

$ echo "${AA[@]}"
Arch Linux Ubuntu Fedora AIX HP-UX

$ BB=( 11 22 33 )
$ echo ${#BB[@]}
3

$ BB+=( 44 )
$ echo ${#BB[@]}
4

$ declare -A AA=( [aa]=11 [bb]=22 [cc]=33 )
$ echo ${#AA[@]}
3

$ AA+=( [dd]=44 )
$ echo ${#AA[@]}
4
```

array에 패턴을 적용하여 치환해 보자.

```bash
$ AA=( "Arch Linux" "Ubuntu Linux" "Suse Linux" "Fedora Linux" )

# change first character u to X
$ echo "${AA[@]/u/X}"
Arch LinXx UbXntu Linux SXse Linux Fedora LinXx

# change all characters u to X
$ echo "${AA[@]//u/X}"
Arch LinXx UbXntX LinXx SXse LinXx Fedora LinXx

# Su로 시작하는 word를 공백으로 치환
$ echo "${AA[@]/Su*/}"
Arch Linux Ubuntu Linux Fedora Linux

$ AA=( "${AA[@]/Su*/}" )

$ echo ${#AA[@]}                    # 원소개수가 4 개로 그대로다.
4

$ for v in "${AA[@]}"; do echo "$v"; done
Arch Linux
Ubuntu Linux
                                   # index 2 는 공백으로 나온다.
Fedora Linux
```

array에 패턴을 적용하여 삭제해 보자.

```bash
$ AA=( "Arch Linux" "Ubuntu Linux" "Suse Linux" "Fedora Linux" )

# "${AA[*]}" 는 "elem1Xelem2Xelem3X..." 와 같습니다.
# 그러므로 IFS 값을 '\n' 바꾸고 echo 한것을 명령치환 값으로 보내면 
# '\n' 에의해 원소들이 분리되어 array 에 저장되게 됩니다.

$ set -f; IFS=$'\n'
$ AA=( $(echo "${AA[*]/Su*/}") )
$ set +f; IFS=$' \t\n'            # array 입력이 완료되었으므로 IFS 값 복구

$ echo ${#AA[@]}                  # 삭제가 반영되어 원소개수가 3 개로 나온다
3

$ echo "${AA[1]} ${AA[2]}"        # index 도 정렬되었다.
Ubuntu Linux Fedora Linux
```

문자열에서 특정 문자를 구분자로 하여 필드 구분하기

```bash
$ AA="Arch Linux:Ubuntu Linux:Suse Linux:Fedora Linux"

$ IFS=: read -ra ARR <<< "$AA"
$ echo ${#ARR[@]}
4
$ echo "${ARR[1]}"
Ubuntu Linux

# 입력되는 원소값에 glob 문자가 있을경우 globbing 이 발생할수 있으므로 noglob 옵션 설정
$ set -f; IFS=:             # IFS 값을 ':' 로 설정
$ ARR=( $AA )
$ set +f; IFS=$' \t\n'      # array 입력이 완료되었으므로 IFS 값 복구

$ echo ${#ARR[@]}
4
$ echo "${ARR[1]}"
Ubuntu Linux

# array 를 사용할 수 없는 sh 에서는 다음과 같이 할 수 있습니다.
$ set -f; IFS=:                   # globbing 을 disable
$ set -- $AA                      # IFS 값에 따라 원소들을 분리하여 
$ set +f; IFS=$' \t\n'            # positional parameters 에 할당

$ echo $#
4
$ echo "$2"
Ubuntu Linux
```

파일을 읽어서 array에 저장하기

```bash
$ cat a.txt 
100  Emma    Thomas
200  Alex    Jason
300  Madison Randy

$ mapfile -t arr < a.txt 

$ echo "${arr[0]}"
100  Emma    Thomas

$ echo "${arr[1]}"
200  Alex    Jason
```

array index에서 arithmatic expression을 해보자.

```bash
$ AA=(1 2 3 4 5)
$ idx=2
$ echo "${AA[ idx == 2 ? 3 : 4 ]}"
4
```

array index의 `[]`와 globbing의 `[]`는 같다. 다음과 같은 문제가 발생할 수 있다.

```bash
$ array=( [10]=100 [11]=200 [12]=300 )
$ echo ${array[12]}
300

$ touch array1               # 현재 디렉토리에 임의로 array1 파일생성

# unset 을 실행하였으나 globbing 에의해 array[12] 이 array1 파일과 매칭이되어 
# 실제적으로 unset -v array1 명령과 같게되어 unset 이 되지 않고 있습니다.
$ unset -v array[12]         
$ echo ${array[12]}         
300                       

$ unset -v 'array[12]'       # globbing 을 disable 하기위해 quote.
$ echo ${array[12]}          # 이제 정상적으로 unset 이됨
```

assignment operator는 globbing 이전에 처리된다.

```bash
$ touch array2

# 대입연산은 globbing 이전에 처리되므로 정상적으로 대입된다.
$ array[12]=100

$ echo ${array[12]}
100
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
# Test

commandline 에 사용된 arguments는 모두 문자열이다.

```bash
# "77"과 77은 둘다 문자열로 취급된다.
$ printf "%d, %s\n" "77" 77

# 문자를 만들어서 expr에 전달한다.
$ expr "-16" + 10
-6

# '*' glob character이기 때문에 escape이 필요하다.
$ echo "10" + 2 \* 5 | bc
20

# 25를 사용하거나 "25"를 사용해도 결과는 같다.
$ [ "150" -gt 25 ]; echo $?
0
$ [ "150" -gt "25" ]; echo $?
0

$ AA=123
$ case $AA in (123) echo Y ;; esac
Y
$ case $AA in ("123") echo Y ;; esac
Y
```

bash는 데이터타입이 없다. operator중 숫자를 취급하는 녀석과 문자열을 취급하는 녀석이 별도로 있다. 다음은 문자열을 취급하는 operator이다.

```bash
-z, -n, =, !=, < >
```

다음은 숫자를 취급하는 operator이다.

```bash
  -eq, -ne, -lt, -le, -gt, or -ge.
```

test expression은 `test`와 `[` command를 이용한다. `]`는 `[`의
argument이다.

```bash
$ test -d /home/iamslash/tmp; echo $?
$ [ -d /home/iamslash/tmp ]; echo $?
```

`[` command 에서 operator를 사용하지 않을 경우 비어있거나 null인 경우는 false이고 나머지는 true이다.

```bash
# false
$ [ ]; echo $?
$ [ "" ]; echo $?
$ [ $foo ]; echo $?
$ A=""
$ [ $A ]; echo $?
# true
$ [ 0 ]; echo $?
$ [ 1 ]; echo $?
$ [ a ]; echo $?
$ [ -n ]; echo$?
```

bash에서 `true`, `false`는 builtin command이다. `true`는 0을 리턴하고 `false`는 1을 리턴한다.

```bash
$ [ true ]; echo $?
0
$ [ false ]; echo $?
0
$ if false; then echo true; else echo false; fi
false
$ foo=true
$ if $foo; then echo true; else echo false; fi
true
```

test expression에서 사용되는 변수는 주로 quote해서 사용해야 오류를 방지할 수 있다.

```bash
# A는 null이다.
$ A=""
$ [ -n $A ]; echo $? # $A가 null인데 true이다. 비정상.
0
$ [ -n "$A" ]; echo $? # $A가 null이므로 false이다. 정상
1

$ A="foo bar"
$ if [ -n $A ]; then echo "$A"; fi
-bash: [: foo: binary operator expected
$ A="foo"
$ if [ -n $A ]; then echo "$A"; fi
foo
$
```

숫자를 연산할 때 string operator를 사용하면 안된다.

```bash
$ [ 100 \> 2]; echo $?
1
$ [ 100 -gt 2 ]; echo $?
0
```

test expression의 경우 `-a`, `o`대신 `&&`, `||` 를 사용할 수도 있다. `{ ;}`를 함께 사용하여 연산자 우선순위를 조절할 수도 있다.

```bash
$ if [ ... ] && [ ... ]; then ...
$ if [ ... ] || [ ... ]; then ...
$ if [ ... ] || { [ ... ] && [ ... ] ;} then ...
$ if [ ... -a ... ]; then ...
$ if [ ... -o ... ]; then ...
# 우선순위 조절을 위해 사용한 ()는 escape이 필요하다.
$ if [ ... -a \( ... -o ... \) ]; then ...
$ if [ ! ... ]; then ...
$ if ! [ ... ]; then ...
```

quote하지 않을 경우 `$[arr[@]]`와 `$[arr[*]]`가 같다. quote할 경우 `"$[arr[@]]"`는 원소들을 `" "`를 구분자로 나열하고 `"$[arr[*]]"`는 원소들을 `" "`안에 삽입한다. 따라서 array를 비교할 경우 `"$[arr[*]]"`를 사용한다.

```bash
$ A=(11 22 33)
$ B=(11 22 33)
$ [ "${A[*]}" = "${B[*]}" ]; echo $?
0
$ [ "${A[@]}" = "${B[@]}" ]; echo $?
error
```

test expression은 `{;}`, `()`, `;`, `|`을 이용하여 다수의 command 가 존재할 수 있다. 가장 마지막에 실행되는 command의 결과값이 test expression의 결과값이 된다.

```bash
until read -r line
  line=$(tr -d `\r\n` <<< "$line")
  test -z "$line"
do
  ...
done
```

`[[ ]]`는 `[ ]`의 확장이다. `[ ]`는 command이고 `[[ ]]`는 keyword이다. keyword는 command처럼 escape하거나 quote할 필요가 없다.

```bash
$ [[ a < b ]]; echo $?
0
$ [[ a > b ]]; echo $?
1
$ A=""
$ [[ -n $AA ]]; echo $?
1
$ [[ -n "$AA" ]]; echo $?
1
```

# Special Expressions

`$(( ))`, `(( ))`는 arithmatic expression을 처리할 수 있다. `[[ ]]`는 text expression을 처리할 수 있다. command와 달리 excape, quote가 필요 없다. command의 경우 참은 0이고 거짓은 1이다. 그러나 arithmatic expression의 경우 참은 1이고 거짓은 0이다. 

```bash
$ A=10
$ (( A = $A + A))
$ echo $A
20
$ A=(5 6 7)
$ (( A[0] = A[1] + ${#A[@]} ))
$ echo ${A[0]}
9

$ (( 1 < 2 )); echo $?
0
$ (( 1 > 2 )); echo $?
1
$ (( 0 )); echo $?
1
$ (( 1 )); echo $?
0
$ (( var = 0, res = (var ? 3 : 4) ))
$ echo $res
4
$ (( var = 1, res = (var ? 3 : 4) ))
$ echo $res
3
$ A=10
$ while (( A > 5 )); do
  echo $A
  (( A-- ))
done
```

`$(( ))`는 `$()` command substitution과 같이 연산의 결과를 저장할 수 있다. `:` builtin command는 arguments expanding및 redirection을 제외하고 아무것도 하지 않는다.

```bash
$ A=$(( 1 + 2 )); echo $A
3
$ expr 100 + $(( A + 2 ))
105
$ $(( 1 + 2 ))
3: command not found
# ':' command 를 사용하면 실행가능하다.
$ : $(( A+= 1 ))
$ echo $A
4
```

`for`와 함께 사용할 수 있다.

```bash
for (( i = 0; i < 10; i++ ))
do
  echo $i
done

for (( i = 0; i < 10; i++ )) {
    echo $i
}
```

`$(( ))`, `(( ))`는 간단한 식을 작성하기에 번거롭다. 그래서 `let`이 등장했다.

```bash
(( i++ ))
let i++
res=$(( var + 1 ))
let res=var+1
```

`let`은 arithmatic하나당 operator하나를 갖는다. 공백에 유의하자. quote를 하면 공백과 함께 사용할 수도 있다.

```
$ let var = 1 + 2
error
$ let var += 1
error
$ let var=1+3
$ let "var++" "res = (var == 5 ? 10 : 20)"; echo $res
200
$ let "2 < 1"; echo $?
1
$ help let
```

# Sub Shell

`( )`, `$( )`, ` `, `|`, `&` 으로 command 실행하면 만들어지는 shell을 subshell이라고 한다.

```
$ ( sleep 10; echo )
$ `sleep 10; echo`
$ echo | { sleep 10; echo;}
$ command &
```

child process가 parent process로 부터 다음과 같은 것들을 물려 받는다.

* 현재 디렉토리
* export된 환경 변수, 함수
* 현재 설정되어 있는 file descriptor (STDIN(1), STDOUT(2), STDERR(3))
* ignore된 신호 ( trap " INT)

```bash
$ echo "PID : $$, PPID : $PPID"
PID : 24362, PPID : 24309
$ pwd
/root
$ foo=100 bar=200
$ export foo
$ f1() { echo "I am exported function" ;}
$ f2() { echo "I am not exported function" ;}
$ export -f f1
$ trap '' SIGINT
$ trap 'rm -f /tmp/tmpfile' SIGTERM
$ tty
/dev/pts/0
```

* a.sh

```bash
#!/bin/bash
echo "PID : $$, PPID : $PPID"
pwd
echo "foo : $foo"
echo "bar : $bar"
f1
f2
trap
ls -l /proc/$$/fd
```

```bash
$ ./test.sh
PID : 24434, PPID : 24362
/tmp
foo : 100
bar :
I am exported function
a.sh: line 7: f2: command not found
trap -- '' SIGINT
total 0
lrwx------ 1 root root 64 Feb 28 11:55 0 -> /dev/pts/0
lrwx------ 1 root root 64 Feb 28 11:55 1 -> /dev/pts/0
lrwx------ 1 root root 64 Feb 28 11:55 2 -> /dev/pts/0
lr-x------ 1 root root 64 Feb 28 11:55 255 -> /tmp/a.sh
```

parent process에서 설정한 변수, 함수는 export해야 child process에서 사용할 수 있다. 그러나 subshell에서는 export하지 않아도 사용할 수가 있다.

```bash
$ A=100
$ ( echo A value = "$A" )
A value = 100
```

현재 shell에서 사용중인 변수를 subshell에서 변경해봐야 현재 shell에서 적용되지 않는다.

```bash
$ A=100
$ ( A=200; echo A value = "$A" )
A value = 200
$ echo "$A"
100
```

subshell을 생성하여 사용한 변수, 환경 설정 변경은 subshell이 종료되면 사라진다.

```bash
$ echo -n "$IFS" | od -a
0000000  sp  ht  nl
$ ( IFS=:; echo -n "$IFS" | od -a )
0000000   :
$ echo -n "$IFS" | od -a
0000000  sp  ht  nl
$ [ -o noglob ]; echo $?
1
$ ( set -o noglob; [ -o noglob ]; echo $? )
0
# subshell이 종료후 이전 상태로 복귀
$ [ -o noglob ]; echo $?

$ set -- 11 22 33
$ echo "$@"
11 22 33
$ ( set -- 44 55 66; echo "$@" )
44 55 66
$ echo "$@"
11 22 33

$ ulimit -c
0
$ ( ulimit -c unlimited; ulimit -c; ... )
unlimited
$ ulimit -c
0

$ ( export LANG=ko_KR.UTF-8 join -j 1 -a 1 <(sort a.txt) <(sort b.txt) )

$ pwd
$ ( cd ~/tmp2; pwd; ... )
$ pwd

$ ( echo hello; exit 3; echo world )
hello
$ echo $?
3

# $$는 현재 shell의 PID이다. subshell에서는 #BASHPID를 사용하자.
$ echo $$ $BASHPID
1111 1111
$ ( echo $$ $BASHPID )
1122 2211

```

# Job Control

## Job Control Builtins

```bash
bg, fg, jobs, kill, wait, disown, suspend
```
# Signals

## signal list

```bash
$ kill -l
 1) SIGHUP       2) SIGINT       3) SIGQUIT      4) SIGILL       5) SIGTRAP
 2) SIGABRT      7) SIGBUS       8) SIGFPE       9) SIGKILL     10) SIGUSR1
1)  SIGSEGV     12) SIGUSR2     13) SIGPIPE     14) SIGALRM     15) SIGTERM
2)  SIGSTKFLT   17) SIGCHLD     18) SIGCONT     19) SIGSTOP     20) SIGTSTP
3)  SIGTTIN     22) SIGTTOU     23) SIGURG      24) SIGXCPU     25) SIGXFSZ
4)  SIGVTALRM   27) SIGPROF     28) SIGWINCH    29) SIGIO       30) SIGPWR
5)  SIGSYS      34) SIGRTMIN    35) SIGRTMIN+1  36) SIGRTMIN+2  37) SIGRTMIN+3
6)  SIGRTMIN+4  39) SIGRTMIN+5  40) SIGRTMIN+6  41) SIGRTMIN+7  42) SIGRTMIN+8
7)  SIGRTMIN+9  44) SIGRTMIN+10 45) SIGRTMIN+11 46) SIGRTMIN+12 47) SIGRTMIN+13
8)  SIGRTMIN+14 49) SIGRTMIN+15 50) SIGRTMAX-14 51) SIGRTMAX-13 52) SIGRTMAX-12
9)  SIGRTMAX-11 54) SIGRTMAX-10 55) SIGRTMAX-9  56) SIGRTMAX-8  57) SIGRTMAX-7
10) SIGRTMAX-6  59) SIGRTMAX-5  60) SIGRTMAX-4  61) SIGRTMAX-3  62) SIGRTMAX-2
11) SIGRTMAX-1  64) SIGRTMAX
```

## kill

다음은 `kill`을 이용하여 `SIGTERM`을 보내는 방법이다.

```bash
$ kill -TERM 1111
$ kill -SIGTERM 1111
$ kill -s TERM 1111
$ kill -s SIGTERM 1111
$ kill -15 1111
$ kill -n 15 1111
$ kill 1111
```

`-`를 이용하면 process groupd에 signal을 보낼 수 있다.

```bash
$ kill -TERM -1111
$ kill -- -1111
```

`kill -0`은 process가 살아있고 signal을 보낼 수 있는지 검사한다.

```bash
# 1111 프로세스는 존재할 때
$ kill -0 1111; echo $?
0
# 1111 프로세스는 존재하지 않을때
$ kill -0 1111; echo $?
1
# 0 process에게 signal을 보낼 권한이 없다.
$ kill -9 1; echo $?
1
```

자신의 그룹에 속한 모든 프로세스를 종료해보자.

```bash
$ kill -TERM 0
$ kill 0
$ kill -INT 0
```

## trap

signal handler를 등록하자.

```bash
$ trap 'myhandler' INT
$ myhandler() { ...;}
# default handler로 reset하자.
$ trap INT
$ trap - INT
# signal을 ignore하자.
$ trap '' INT
```

`SIGKILL`, `SIGSTOP`, `SIGCONT`는 trap으로 handler를 등록할 수 없다. default handler만 사용 가능하다.

process가 정상종료될 때 handler를 등록하려면 `HUP, INT, QUIT, TERM`등의 signal을 trap해야 한다. 그러나 `EXIT`라는 pseudo signal을 하나만 등록해도 동일한 기능을 한다. 다음은 pseudo signal의 목록이다.

| Signal | Description |
|-----|-----|
| EXIT | shell 이 exit 할때 발생. ( subshell 에도 적용 가능 ) |
| ERR |	명령이 0 이 아닌 값을 리턴할 경우 발생. |
| DEBUG |	명령 실행전에 매번 발생. |
| RETURN |	함수에서 리턴할때, source 한 파일에서 리턴할때 발생. |

```bash 
$ trap 'myhandler' HUP INT QUIT TERM
$ myhandler() { ...;}
```

# Problems

* [Word Frequency](https://leetcode.com/problems/word-frequency/description/)

```bash
cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{ print $2, $1}'
```

* [Transpose File](https://leetcode.com/problems/transpose-file/description/)

```bash
awk '
{ 
    for (i = 1; i <= NF; i++) {
        if (NR == 1) {
            s[i] = $i;
        } else {
            s[i] = s[i] " " $i;
        }
    }
}
END {
    for (i = 1; s[i] != ""; ++i) {
        print s[i];
    }
}' file.txt
```