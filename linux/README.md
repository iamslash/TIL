# Abstract

linux를 활용할 때 필요한 지식들을 정리한다. macosx, sunos, hp-ux 등등
unix 계열 os는 모두 해당된다.

# References

* [Most Important Penetration Testing commands Cheat Sheet for Linux Machine](https://techincidents.com/important-penetration-testing-cheat-sheet/)
* [command line reference](https://ss64.com/)
  * bash, macOS, cmd, powershell 등등의 command line reference

# Permissions

## mode

`chmod`의 첫번째 인자가 mode이고 8진법으로 4자리이다.

```bash
chmod 4000 a.txt
```

mode는 8진법으로 표기했을때 왼쪽 부터 특수권한, 유저권한, 그룹권한,
기타권한과 같이 구성된다.  각 권한 별로 3비트가 할당된다. 특수권한의
3비트는 왼쪽부터 setuid, setgid, stckybit을 의미하고 유저권한,
그룹권한, 기타권한의 3비트는 왼쪽부터 읽기, 쓰기, 실행 권한을
의미한다.

특수권한을 확인 하는 요령은 다음과 같다. `ls -l` 했을때 setuid가
on되어 있으면 유저권한의 3비트중 가장 오른쪽 비트가 s 혹은 S로
보여진다.  setgid가 on되어 있으면 그룹권한의 3비트중 가장 오른쪽
비트가 s 혹은 S로 보여진다.  stickybit가 on되어 있으면 기타권한의
3비트중 가장 오른쪽 비트가 t 혹은 T로 보여진다.  표시되는 권한의
실행권한이 없다면 소문자로 보여지고 실행권한이 있다면 대문자로
보여진다.

### setuid

setuid가 설정된 파일을 실행할때 다음과 같은 현상이 발생한다.  실행을
위해 태어난 프로세스의 EUID(유효 사용자 아이디)가 RUID(실행 사용자
아이디)에서 파일의 소유자 아이디로 변경된다.

실행순간만 권한을 빌려온다고 이해하자.

### setgid

setgid가 설정된 파일을 실행할때 다음과 같은 현상이 발생한다.  실행을
위해 태어난 프로세스의 EGID(유효 그룹 아이디)가 RGID(실행 그룹
아이디)에서 파일의 소유 그룹 아이디로 변경된다.

실행순간만 권한을 빌려온다고 이해하자.

### sticky bit

linux는 파일의 sticky bit를 무시한다. 디렉토리에 sticky bit가 설정되어
있다면 누구나 해당 디렉토리에서 파일을 생성할 수 있지만 삭제는
디렉토리 소유자, 파일 소유자, 슈퍼 유저만 할 수 있다. 그래서 sticky bit를
공유모드라고 한다.

# Special Directories

| DIRECTORY | DESCRIPTION |
|-----------|-------------|
| /	    | / also know as “slash” or the root. |
| /bin	| Common programs, shared by the system, the system administrator and the users. |
| /boot	| Boot files, boot loader (grub), kernels, vmlinuz |
| /dev	| Contains references to system devices, files with special properties. |
| /etc	| Important system config files. |
| /home	| Home directories for system users. |
| /lib	| Library files, includes files for all kinds of programs needed by the system and the users. |
| /lost+found	| Files that were saved during failures are here. |
| /mnt	| Standard mount point for external file systems. |
| /media|	Mount point for external file systems (on some distros). |
| /net	| Standard mount point for entire remote file systems ? nfs. |
| /opt	| Typically contains extra and third party software. |
| /proc	| A virtual file system containing information about system resources. |
| /root	| root users home dir. |
| /sbin	| Programs for use by the system and the system administrator. |
| /tmp	| Temporary space for use by the system, cleaned upon reboot. |
| /usr	| Programs, libraries, documentation etc. for all user-related programs. |
| /var	| Storage for all variable files and temporary files created by users, such as log files, mail queue, print spooler. Web servers, Databases etc. |

# Special Files

| DIRECTORY	| DESCRIPTION |
|-----------|-------------|
| /etc/passwd   |  Contains local Linux users. |
| /etc/shadow   |	Contains local account password hashes. |
| /etc/group    |	Contains local account groups. |
| /etc/init.d/  |	Contains service init script ? worth a look to see whats installed. |
| /etc/hostname	| System hostname. |
| /etc/network/interfaces |	Network interfaces. |
| /etc/resolv.conf	      | System DNS servers. |
| /etc/profile	          | System environment variables. |
| ~/.ssh/	| SSH keys. |
| ~/.bash_history	| Users bash history log. |
| /var/log/	        | Linux system log files are typically stored here. |
| /var/adm/	        | UNIX system log files are typically stored here. |
| /var/log/apache2/access.log | Apache access log file typical path. |
| /var/log/httpd/access.log | Apache access log file typical path. |
| /etc/fstab	| File system mounts. |

# Package Managers
## apt-get
## brew
## yum
# Commands

application commands와 bash builtin commands등이 있다.  상황별로
유용한 commands를 정리한다. bash builtin commands의 경우 `/usr/bin/`
에 application commands으로 존재한다. 다음은 macosx에서
`/usr/bin/ulimit`의 내용이다. 단지 bash builtin으로 command와
argument들을 전달 하고 있다.


```bash
builtin `echo ${0##*/} | tr \[:upper:] \[:lower:]` ${1+"$@"}
```

## 메뉴얼

* `man`
  * 메뉴얼 좀 보여줘봐
  * `man ls`
* `apropos`
  * 잘은 모르겠고 이런거 비슷한 거 찾아줘봐라
  * `man -k`와 같다.
  * `apropos brew`
* `info`
  * 메뉴얼 좀 보여줘봐. 단축키는 emacs와 비슷한데?
  * `info ls`

## 자주 사용

* `history`
  * 최근에 사용한 command line보여줘봐라
  * `history` `!ssh` `!!` `!14`
* `ls`
  * 디렉토리들과 파일들을 보여줘라.
  * `ls -al`
* `cd`
  * 작업디렉토리를 바꿔보자.
  * `cd /usr/bin`
  * `cd ~`
* `pwd`
  * 현재 작업디렉토리는 무엇이냐
* `pushd, popd`
  * 디렉토리를 스택에 넣고 빼자.
  * `pushd /usr/bin` `cd` `cd popd`
* `ln`
  * 심볼릭 링크좀 만들자
  * `ln -s /Users/iamslash/demo /Users/iamslash/mylink`
* `cp`
  * 복사 좀 하자
  * `cp -r a b`
* `mv`
  * 파일을 옮기거나 파일 이름 바꾸자
  * `mv a b`
* `rm`
  * 좀 지워 볼까?
  * `rm -rf *`
* `cat`
  * 파일을 이어 붙이거나 출력하자
  * `cat a.txt`
  * `cat a.txt b.txt > c. txt`
* `more`
  * 한번에 한화면씩 출력한다.
  * `man ls | more`
* `less`
  * `more`보다 기능이 확장 된 것
  * `man ls | less`
* `echo`
  * 화면에 한줄 출력하자
  * `echo hello` `echo $A`
* `touch`
  * 파일의 최종 수정 날짜를 바꾸자
  * `touch a.txt`
* `diff`
  * 두개의 파일들을 줄 단위로 비교하자.
  * `diff a.txt b.txt`
* `which`
  * command위치는 어디있어?
  * `which ls`
* `file`
  * 이 파일의 종류는 무엇이지?
  * `file a.out`
* `ps`
  * 현재 실행되고 있는 프로세스들의 스냅샷을 보여다오
  * `ps aux`
    * BSD syntax로 보여다오
  * `ps axjf`
    * 트리형태로 보여다오
  * `ps axms`
    * 쓰레드들 보여다오
* `kill`
  * 번호를 이용하여 특정 프로세스에게 시글널을 보내자.
  * `kill -l`
    * 가능한 시글널 목록을 보여다오
  * `kill -9 123`
    * 123 프로세스에게 SIGKILL보내다오
* `killall`
  * 이름을 이용하여 특정 프로세스에게 시그널을 보내자.
  * `killall -9 a.sh`
* `pgrep`
  * 프로세스 번호를 프로세스 이름을 grep하여 찾아다오
  * `pgrep -u root aaa`
    * root가 소유하고 aaa라는 이름의 프로세스 번호를 찾아다오 
* `pkill`
  * 이름에 해당하는 프로세스에게 시그널을 보내자.
  * `pkill -HUP syslogd`
* `pstree`
  * 프로세스의 트리를 보여다오
  * `pstree -a`
* `telnet`
  * TELNET protocol client
* `nc`
  * 방화벽이 실행되고 있는지 확인하기 위해 특정 포트에 리슨해 보자.
  * `nc -l 1234`
    * 1234포트에 리슨해보자.
  * `nc 127.0.0.1 1234`
    * 1234포트로 접속해보자.
* `ssh`
  * OpenSSH SSH client
  * `ssh iamslash@a.b.com`
  * `ssh iamslash@a.b.com ls /tmp/doc`
* `sftp`
  * secure file transfer program
  * `sftp iamslash@a.b.com` `cd a` `lcd a` `put a.txt` 
* `scp`
  * secure remote copy program
  * `scp iamslash@a.b.com:a.txt ~/tmp`
  * `scp a.txt iamslash@a.b.com:/home/iamslash/tmp`
* `ftp`
  * file transfer program
* `screen, tmux`
  * [tmux](../tmux/)
  * terminal multiplexer
* `nslookup`
  * domain을 주고 ip로 확인하자.
  * `nslookup www.google.co.kr`
* `dig`
  * ip를 주고 domain을 확인하자.
  * `dig -x 216.58.200.3`
* `curl`
  * URL을 활용하여 data전송하는 program. HTTP, HTTPS, RTMP등등을 지원한다.
  * `curl "http://a.b.com/a?a=1"`
  * `curl --data "a=1&b=%20OK%20" http://a.b.com/a`
  * `curl --data-urlencode "a=I am david" http://a.b.com/a`
* `wget`
  * web에서 파일좀 내려받아다오
  * `wget ftp://a.b.com/a.msg`
* `traceroute`
  * 네트워크 호스트까지 경로를 추적하자. 특정 라우터 까지 어떤 라우터들을 거쳐 가는가?
  * `traceroute www.google.com`
* `locate, updatedb`
  * 파일이름에 해당하는 걸 찾아다오
* `sudo`
  * 다른 유저로 command를 실행하자.
  * `sudo find / -name "aaa"`
* `su`
  * EUID, EGID를 수정하여 SHELL을 실행하자.
  * `su - root`
* `bc`
  * 계산기 언어
  * `echo "56.8 + 77.7" | bc` 
  * `echo "scale=6; 60/7.02" | bc`
* `reset`
  * 터미널을 초기화 한다. 지저분 해졌을때 사용하면 좋음
* `tee`
  * stdin으로 입력 받고 stdout과 파일로 출력하자.
  * `ls | tee a.txt`
    * `ls > file`은 stdout말고 파일로만 출력한다.
  * `ls | tee -a a.txt`
  * `ls | tee a.txt b.txt c.txt`
* `script`
  * 갈무리
  * `script a.txt` `exit`
 
## 유저 관리

* `useradd`
  * `useradd iamslash`
* `passwd`
  * `passwd iamslash`
* `deluser`
  * `deluser iamslash`

## 파일 권한

* `chmod`
  * `chomod -R 777 tmp`
* `chown`
  * `chown -R iamslash tmp`
* `chgrp`
  * `chgrp -R staff tmp`

## 시스템 모니터링

* `netstat`
* `ss`
* `top`
* `htop`
* `atop`
* `lsof`
* `lshw`
* `vmstat`
* `who`
* `ifconfig`
* `date`
* `time`
* `du`
* `df`
* `uname`
* `hostname`
* `last`
* `uptime`
* `ping`

## 로그

* `grep`
* `xargs`
* `find`
  * `find . -name "*.log" | xargs cat`
* `tail`
* `head`
* `awk`
* `uniq`
* `sort`
* `wc`
  * `wc -l`

## 텍스트

* `paste`
* [sed](../sed/)
* `tr`
* `printf`
* `nl`
* `seq`
* `sha1sum`

## 디버깅

* `gdb`
* `ldd`
  * shared library 의존성을 알려다오
  * `ldd execv`
  * `ldd a.so`
  * `ldd -v a.so`
  * `ldd -u func` unused direct dependencies
  * `ldd /bin/ls` ldd wants absolute path
* `nm`
  * symbol table을 알려다오
  * `nm  -A ./*.o | grep func` func를 포함한 심볼들을 알려주라
  * `nm -u a.out` 사용하지 않거나 실행시간에 share libary에 연결되는 심볼들을 알려주라
  * `nm -n a.out` 모든 심볼들을 알려주라
  * `nm -S a.out | grep abc` abc를 포함한 심볼을 크기와 함께 알려다오
  * `nm -D a.out` dynamic 심볼좀 알려주라
  * `nm -u -f posix a.out` bsd형식 말고 posix형식으로 알려다오
  * `nm -g a.out` external 심볼만 알려다오
  * `nm -g --size-sort a.out` 실볼을 크기 정렬해서 알려다오
* `strace`
  * system call과 signal을 모니터링 해주라. 소스는 없는데 디버깅 하고 싶을때 유용하다
  * `strace ls`
  * `strace -e open ls` system call중 open만 보여주라
  * `strace -e trace-open,read ls /home` system call중 open, read보여주라
  * `strace -o a.txt ls` `cat a.txt`
  * `ps -C firefox-bin` `sudo strace -p 1725 -o firefox_trace.txt` `tail -f firefox_trace.txt`
  * `strace -p 1725 -o output.txt`
  * `strace -t -e open ls /home` 절대 시간으로 보여다오
  * `strace -r ls` 상대 시간으로 보여다오
  * `strace -c ls /home` 테이블 형태로 보여다오
* `ltrace`
  * library call좀 보여다오
  * `ltrace ls`
  * `ltrace -p 1275`
* `pstack`
  * process의 callstack을 thread별로 보여다오
  * `pstack 1275` 
* `pmap`
  * process의 memory map좀 보여다오
  * `pmap 1275`
  * `pmap -x 1275`
  * `pmap -d 1275`
  * `pmap -q 1275` header와 footer는 보여주지 말아라
  * 
* `valgrind`
  * 메모리 누수를 검사하자.
  * `valgrind --leak-check=yes myprog arg1 arg2`
* `od`
  * octal numbers(8진수)로 보여다오
  * `od -b a.txt`
  * `od -c a.txt` 캐릭터로 보여줄래?
  * `od -Ax -c a.txt` byte offset을 hexadecimal형식으로 보여다오
  * `od -An -c a.txt` byte offset제외 하고 보여다오
  * `od -j9 -c a.txt` 9bytes  건너뛰고 보여다오
  * `od -N9 -c a.txt` 9bytes만 보여다오
  * `od -i a.txt` decimal 형식으로 보여다오
  * `od -x a.txt` hexadecimal 2 byte단위로 보여다오
  * `od -o a.txt` octal 2 byte단위로 보여다오
  * `od -w1 -c -Ad a.txt`
  * `od -w1 -v -c -Ad a.txt` 중복된 것도 보여줘
* `strings`
  * 최소한 4byte보다 크거나 같은 문자열을 보여다오
  * `strings a.out`
  * `strings a.out | grep hello`
  * `strings -n 2 welcome | grep ls`
  * `strings -o a.out` offset도 알려줘
  * `strings -f /bin/* | grep Copy` 여러개의 파일들의 스트링을 알려줘

## 압축

* `tar`
  * `tar -cvf a.tar /home/iamslash/tmp`
  * `tar -cvzf a.tar.gz /home/iamslash/tmp`
  * `tar -cvjf a.tar.gz2 /home/iamslash/tmp`
  * `tar -xvf a.tar`
  * `tar -xvf a.tar.gz`
  * `tar -xvf a.tar.gz2`
* `compress`
  * Lempel-Ziv 코딩으로 압축한다.
  * `compress a.txt`
  * `uncompress a.txt.Z`
* `gzip`
  * `gzip a.txt`
  * `gzip a.txt b.txt c.txt`
  * `gzip -c a.txt > a.txt.gz` a.txt를 지우지 않는다.
  * `gzip -r *`
  * `gzip -d a.txt.gz`
  * `gunzip a.txt.gz`
  * `gunzip -c a.txt.gz > a.txt` a.txt.gz를 지우지 않는다.
  * `zcat a.txt.gz`
  * `zgrep exa a.txt.gz`
* `bzip2`
  * gzip보다 압축률이 좋다.
  * `bzip2 a.txt`
  * `bzip2 a.txt b.txt c.txt`
  * `bzip2 -c a.txt > a.txt.bz2` a.txt를 지우지 않는다.
  * `bzip2 -d a.txt.bz2`
  * `bunzip2 -c a.txt.bz2 > a.txt`
  * `time bzip2 -v -1 a.tar` `time bzip2 -v -9 a.tar`
  * `bzip2 -tv a.tar.bz2`
  * `bzip2 -c a.txt > a.bz2` `bzip2 -c b.txt >> a.bz2`
  * `bzcat a.txt.bz2`
  * `bzgrep exa a.txt.bz2`
* `xz`
  * gzip, bzip2보다 압축률이 더욱 좋다.
  * `xz a.txt`
  * `xz a.txt b.txt c.txt`
  * `xz -k a.txt` `xz -c a.txt > a.txt.xz`
  * `xz -d a.txt.xz`
  * `unxz a.txt.xz` `unxz -k a.txt.xz`
  * `xz -l a.tar.xz`
  * `time xz -v a.tar`
  * `time xz -0v a.tar` `time xz -9v a.tar` `time xz -0ve a.tar`
  * `time xz -6ve --threads=0 a.tar` 
  * `tar cJvf a.tar.xz /etc/`
  * `xz -tv a.tar.xz` `xz -t a.tar.xz` `xz -lvv a.tar.xz`
  * `xzip -c a.txt > a.xz` `xzip -c b.txt >> a.xz`
  * `xzcat a.txt.xz` `xzgrep exa a.txt.xz`
* `zless`
  * `zless a.txt.gz`
* `bzless`
  * `bzless a.txt.bz2`

## 에디터

* `vi, vim, emacs, nano, ed`

# Security

## root 소유의 setuid, setgid파일 검색 후 퍼미션 조정하기

setuid가 설정되어 있으면 실행 되었을 때 EUID가 root로 변경된다. 불필요한
파일을 찾아서 퍼미션을 변경하자.

```bash
find / -user root -perm 4000 -print
```
