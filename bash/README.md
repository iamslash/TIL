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

### use a default value

### assign a default value

### use an alternate value

### display error if null or unset

### substring expansion

### search and replace

### case modification

### indirection

## command expansion

## arithmetic expansion

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
