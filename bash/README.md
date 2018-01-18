# Abstract

bash에 대해 정리한다.

# References

* [Bash script](https://www.gitbook.com/book/mug896/shell-script/details)
  * 친절한 한글
* [bash reference manual @ gnu](https://www.gnu.org/software/bash/manual/bash.html)
* [Advanced Bash-Scripting Guide](http://www.tldp.org/LDP/abs/html/index.html)

# Basic Usages

```bash
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

# &&
# execute several commands
make && make install

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

# if
#
# comparison of numbers
# -eq, -ne, -gt, -ge, -lt, -le
# comparison of strings
# =(==), !=, -z(null), -n(not null)
if [ $a -eq $b ]; then
    echo $a
fi

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

# while
while :
do
    echo "foo";
    sleep 1;
done

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

# Shell Parameters

## Positional Parameters

`${N}`처럼 표기한다. N는 paramter의 서수이다.

```bash
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
  * ???
  
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

## filename expansion

파일 이름을 다룰때 `*`, `?`, `[]`는 패턴 매칭과 동일하게 확장된다.
앞서 언급한 문자들을 glob 문자라고 한다. glob 문자가 확장되는 것을
globbing이라고 한다.

```bash
$ ls
a.a a.b a.c a.h

$ ls *.[ch]
a.c a.h

$ ls "*.[ch]"         # quote 을 하면 globbing 이 일어나지 않는다
ls: cannot access *.[ch]: No such file or directory

$ echo *.?
a.a a.b a.c a.h

$ for file in *.[ch]; do
      echo "$file"
done

a.c
a.h
```

## quote removal

# Shell Variables

사전에 예약된 변수들을 정리한다.

* CDPATH
  * 현재 작업디렉토리
  * `echo $CDPATH` 
* HOME
  * 사용자의 집디렉토리
  * `echo $HOME`
* IFS
  * 파라미터 구분자
  * `echo $IFS`
