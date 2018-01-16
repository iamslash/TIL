# Abstract

linux를 활용할 때 필요한 지식들을 정리한다. macosx, sunos, hp-ux 등등
unix 계열 os는 모두 해당된다. 예제들은 macosx에서 실험했다. ubuntu와
다를 수도 있다.

# References

* [Most Important Penetration Testing commands Cheat Sheet for Linux Machine](https://techincidents.com/important-penetration-testing-cheat-sheet/)

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
`/usr/bin/ulimit`의 내용이다. `${0##*/}` 과 `${1+"$@"}`는 parameter
expansion을 의미한다.


```bash
builtin `echo ${0##*/} | tr \[:upper:] \[:lower:]` ${1+"$@"}
```

## 메뉴얼

* `man`
* `apropos`
  * `man -k`
* `info`

## 자주 사용

* `history`
* `ls, cd, pwd`
* `pushd, popd`
* `ln, cp, mv, rm`
* `cat, more, less`
* `echo`
* `touch`
* `diff`
* `which`
* `file`
* `ps`
  * `ps aux`
* `kill`
* `killall`
* `pkill, pgrep`
* `pstree`
* `telnet`
* `nc`
  * 방화벽이 실행되고 있는지 확인하기 위해 특정 포트에 리슨해 보자.
  * `nc -l 1234`
    * 1234포트에 리슨해보자.
  * `nc 127.0.0.1 1234`
    * 1234포트로 접속해보자.  
* `ssh`
* `sftp`
* `scp`
* `ftp`
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
* `wget`
* `traceroute`
* `locate, updatedb`
* `sudo`
* `su`
* `bc`
* `reset`
* `tee`
* `script`
 
## 유저 관리

* useradd
* passwd
* deluser

## 파일 권한

* chmod
* chown
* chgrp

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
  * dependency walker
* `nm`
* `strace`
* `ltrace`
* `pstack`
* `pmap`
* `valgrind`
* `od`
* `strings`

## 압축

* `tar`
* `compress`
* `gzip`
* `bzip2`
* `xz`
* `zless`
* `bzless`
* `zcat`
* `bzcat`

## 에디터

* `vi, vim, emacs, nano, ed`

# Security

## root 소유의 setuid, setgid파일 검색 후 퍼미션 조정하기

setuid가 설정되어 있으면 실행 되었을 때 EUID가 root로 변경된다. 불필요한
파일을 찾아서 퍼미션을 변경하자.

```bash
find / -user root -perm 4000 -print
```
