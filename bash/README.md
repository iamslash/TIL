# Abstract

bash에 대해 정리한다.

# References

* [Bash script](https://www.gitbook.com/book/mug896/shell-script/details)
  * 친절한 한글
* [Advanced Bash-Scripting Guide](http://www.tldp.org/LDP/abs/html/index.html)
* [bash reference manual @ gnu](https://www.gnu.org/software/bash/manual/bash.html)

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
expansion, command expansion, arithmetic expansion, word splitting,
filename expansion.

## brace expansion

### string lists

`{string1, string2,..., stringN}`과 같이 사용한다.

```bash
$ echo X{apple, banana, orage, melon}Y
X{apple, banana, orange, melon}Y
```

### ranges

### combining and nesting

## tilde expansion

현재 디렉토리로 확장된다.

```
$ echo ~
/Users/iamslash

$ echo ~iamslash # iamslash 유저 디렉토리를 출력하라.
/Users/iamslash
```

## parameter and variable expansion

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

## filename expansion

## quote removal
