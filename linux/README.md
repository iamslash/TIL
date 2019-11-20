- [Abstract](#abstract)
- [References](#references)
- [Permissions](#permissions)
  - [mode](#mode)
    - [setuid](#setuid)
    - [setgid](#setgid)
    - [sticky bit](#sticky-bit)
- [Special Directories](#special-directories)
- [Special Files](#special-files)
- [Package Managers](#package-managers)
  - [apt-get](#apt-get)
  - [brew](#brew)
  - [yum](#yum)
- [Commands](#commands)
  - [메뉴얼](#%eb%a9%94%eb%89%b4%ec%96%bc)
  - [자주 사용](#%ec%9e%90%ec%a3%bc-%ec%82%ac%ec%9a%a9)
  - [Process management](#process-management)
  - [유저 관리](#%ec%9c%a0%ec%a0%80-%ea%b4%80%eb%a6%ac)
  - [파일 권한](#%ed%8c%8c%ec%9d%bc-%ea%b6%8c%ed%95%9c)
  - [시스템 모니터링](#%ec%8b%9c%ec%8a%a4%ed%85%9c-%eb%aa%a8%eb%8b%88%ed%84%b0%eb%a7%81)
  - [로그](#%eb%a1%9c%ea%b7%b8)
  - [텍스트](#%ed%85%8d%ec%8a%a4%ed%8a%b8)
  - [디버깅](#%eb%94%94%eb%b2%84%ea%b9%85)
  - [압축](#%ec%95%95%ec%b6%95)
  - [에디터](#%ec%97%90%eb%94%94%ed%84%b0)
  - [데몬 관리](#%eb%8d%b0%eb%aa%ac-%ea%b4%80%eb%a6%ac)
  - [Automation](#automation)
  - [oneline commands](#oneline-commands)
- [Security](#security)
  - [root 소유의 setuid, setgid 파일 검색 후 퍼미션 조정하기](#root-%ec%86%8c%ec%9c%a0%ec%9d%98-setuid-setgid-%ed%8c%8c%ec%9d%bc-%ea%b2%80%ec%83%89-%ed%9b%84-%ed%8d%bc%eb%af%b8%ec%85%98-%ec%a1%b0%ec%a0%95%ed%95%98%ea%b8%b0)
- [System Monitoring](#system-monitoring)
  - [swapin, swapout](#swapin-swapout)

-------------------------------------------------------------------------------

# Abstract

linux 를 활용할 때 필요한 지식들을 정리한다. macOS, sunos, hp-ux 등등
unix 계열 os 는 모두 해당된다. linux 와 함께 [bash](/bash/),
[awk](/awk/), [sed](/sed/) 역시 학습이 필요하다.

systemd 가 설치된 [ubuntu docker image](https://hub.docker.com/r/jrei/systemd-ubuntu) 를 이용하여 실습하자.

```bash
$ docker run -d --name systemd-ubuntu --privileged jrei/systemd-ubuntu
$ docker exec -it systemd-ubuntu bash
```

# References

* [The Art of Command Line @ github](https://github.com/jlevy/the-art-of-command-line/blob/master/README-ko.md)
  * 킹왕짱 커맨드 라인
* [Most Important Penetration Testing commands Cheat Sheet for Linux Machine](https://techincidents.com/important-penetration-testing-cheat-sheet/)
  * 유용한 시스템 침입 테스트 커맨드들
* [command line reference](https://ss64.com/)
  * bash, macOS, cmd, powershell 등등의 command line reference

# Permissions

## mode

* [참고](https://eunguru.tistory.com/115)

----

```bash
 > chmod 4644 setuidfile.txt 
 > chmod 2644 setgidfile.txt 
 > chmod 1644 stickyfile.txt 

 > ls -l

-rwSr--r--  1 iamslash  staff     0  8 10 21:16 setuidfile.txt
-rw-r-Sr--  1 iamslash  staff     0  8 10 21:16 setgidfile.txt
-rw-r--r-T  1 iamslash  staff     0  8 10 21:16 stickyfile.txt
```

`chmod` 의 첫번째 인자가 mode 이고 8 진법으로 4 자리이다.

```bash
chmod 4000 a.txt
```

mode 는 8 진법으로 표기했을때 왼쪽 부터 특수권한, 유저권한, 그룹권한,
기타권한과 같이 구성된다.  각 권한 별로 3비트가 할당된다. 특수권한의
3 비트는 왼쪽부터 setuid, setgid, stckybit 을 의미하고 유저권한,
그룹권한, 기타권한의 3비트는 왼쪽부터 읽기, 쓰기, 실행 권한을
의미한다.

특수권한을 확인 하는 요령은 다음과 같다. `ls -l` 했을때 setuid 가
on 되어 있으면 유저권한의 3 비트중 가장 오른쪽 비트가 s 혹은 S 로
보여진다. setgid 가 on 되어 있으면 그룹권한의 3 비트중 가장 오른쪽
비트가 s 혹은 S 로 보여진다.  stickybit 가 on 되어 있으면 기타권한의
3 비트중 가장 오른쪽 비트가 t 혹은 T 로 보여진다.  표시되는 권한의
실행권한이 없다면 소문자로 보여지고 실행권한이 있다면 대문자로
보여진다.

### setuid

setuid 가 설정된 파일을 실행할때 다음과 같은 현상이 발생한다.  실행을
위해 태어난 프로세스의 EUID(유효 사용자 아이디)가 RUID(실행 사용자
아이디)에서 파일의 소유자 아이디로 변경된다.

실행순간만 권한을 빌려온다고 이해하자.

### setgid

setgid 가 설정된 파일을 실행할때 다음과 같은 현상이 발생한다.  실행을
위해 태어난 프로세스의 EGID(유효 그룹 아이디)가 RGID(실행 그룹
아이디)에서 파일의 소유 그룹 아이디로 변경된다.

실행순간만 권한을 빌려온다고 이해하자.

### sticky bit

linux 는 파일의 sticky bit 를 무시한다. 디렉토리에 sticky bit 가 설정되어
있다면 누구나 해당 디렉토리에서 파일을 생성할 수 있지만 삭제는
디렉토리 소유자, 파일 소유자, 슈퍼 유저만 할 수 있다. 그래서 sticky bit 를
공유모드라고 한다.

# Special Directories

| DIRECTORY   | DESCRIPTION                                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| /           | / also know as “slash” or the root.                                                                                                            |
| /bin        | Common programs, shared by the system, the system administrator and the users.                                                                 |
| /boot       | Boot files, boot loader (grub), kernels, vmlinuz                                                                                               |
| /dev        | Contains references to system devices, files with special properties.                                                                          |
| /etc        | Important system config files.                                                                                                                 |
| /home       | Home directories for system users.                                                                                                             |
| /lib        | Library files, includes files for all kinds of programs needed by the system and the users.                                                    |
| /lost+found | Files that were saved during failures are here.                                                                                                |
| /mnt        | Standard mount point for external file systems.                                                                                                |
| /media      | Mount point for external file systems (on some distros).                                                                                       |
| /net        | Standard mount point for entire remote file systems ? nfs.                                                                                     |
| /opt        | Typically contains extra and third party software.                                                                                             |
| /proc       | A virtual file system containing information about system resources.                                                                           |
| /root       | root users home dir.                                                                                                                           |
| /sbin       | Programs for use by the system and the system administrator.                                                                                   |
| /tmp        | Temporary space for use by the system, cleaned upon reboot.                                                                                    |
| /usr        | Programs, libraries, documentation etc. for all user-related programs.                                                                         |
| /var        | Storage for all variable files and temporary files created by users, such as log files, mail queue, print spooler. Web servers, Databases etc. |

# Special Files

| DIRECTORY                   | DESCRIPTION                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| /etc/passwd                 | Contains local Linux users.                                         |
| /etc/shadow                 | Contains local account password hashes.                             |
| /etc/group                  | Contains local account groups.                                      |
| /etc/init.d/                | Contains service init script ? worth a look to see whats installed. |
| /etc/hostname               | System hostname.                                                    |
| /etc/network/interfaces     | Network interfaces.                                                 |
| /etc/resolv.conf            | System DNS servers.                                                 |
| /etc/profile                | System environment variables.                                       |
| ~/.ssh/                     | SSH keys.                                                           |
| ~/.bash_history             | Users bash history log.                                             |
| /var/log/                   | Linux system log files are typically stored here.                   |
| /var/adm/                   | UNIX system log files are typically stored here.                    |
| /var/log/apache2/access.log | Apache access log file typical path.                                |
| /var/log/httpd/access.log   | Apache access log file typical path.                                |
| /etc/fstab                  | File system mounts.                                                 |

# Package Managers

## apt-get

* [apt-get(8) - Linux man page](https://linux.die.net/man/8/apt-get)

```bash
# update the list of packages
apt-get update
# upgrade every packages
apt-get upgrade
apt-get install curl
apt-get --reinstall install curl
apt-get remove curl
# remove curl package including config files
apt-get --purge remove curl
# download curl pakcage source
apt-get source curl
# search curl pakcage
apt-cache search curl
# show curl package informations
apt-cache show curl
```

## brew

```bash
brew install curl
```

## yum

```bash
yum install curl
```

# Commands

* [The Art of Command Line @ github](https://github.com/jlevy/the-art-of-command-line/blob/master/README-ko.md)

<br/>

application commands 와 bash builtin commands 등이 있다.  상황별로
유용한 commands 를 정리한다. bash builtin commands 의 경우 `/usr/bin/`
에 application commands 으로 존재한다. 다음은 macOS 에서
`/usr/bin/ulimit` 의 내용이다. 단지 bash builtin 으로 command 와
argument 들을 전달 하고 있다.

```bash
builtin `echo ${0##*/} | tr \[:upper:] \[:lower:]` ${1+"$@"}
```

## 메뉴얼

* `man`
  * 메뉴얼 좀 보여줘봐
  * `man ls`
* `apropos`
  * 잘은 모르겠고 이런거 비슷한 거 찾아줘봐라
  * `man -k` 와 같다.
  * `apropos brew`
* `info`
  * 메뉴얼 좀 보여줘봐. 단축키는 emacs 와 비슷한데?
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
  * `more` 보다 기능이 확장 된 것
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
  * command 위치는 어디있어?
  * `which ls`
  * `command -v ls`
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
    * 123 프로세스에게 SIGKILL 보내다오
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
  * `apt-get install psmisc`
  * 프로세스의 트리를 보여다오
  * `pstree -a`
* `telnet`
  * TELNET protocol client
* `nc` netcat
  * [넷캣(Netcat) 간단한 사용방법](https://devanix.tistory.com/307)
  * `nc -l 1234`
    * 방화벽이 실행되고 있는지 확인하기 위해 1234 포트에 리슨해보자.
  * `nc 127.0.0.1 1234`
    * 1234포트로 접속해보자.
  * `ps auxf | nc -w3 10.0.2.15 1234`
    * `ps auxf` 결과를 전송하자.
    * `-w3` 를 사용하여 3 초 동안 전송해 보자.
  * `nc -n -v -z -w 1 10.0.2.100 1-1023`
    * `-n` : DNS 이름 주소 변환 안한다.
    * `-v` : verbose
    * `-z` : 데이터 전송 안한다.
    * `-w` : 최대 1 초의 연결 시간
    * `1-1023` : TCP 를 사용해서 1-1023 사이의 포트를 스캔한다.
* `ssh`
  * OpenSSH SSH client
  * `ssh iamslash@a.b.com`
  * `ssh iamslash@a.b.com ls /tmp/doc`
* `ssh-add`
  * 인증 에이전트에게 비밀키를 추가해 놓자.
  * `ssh-add -K ~/.ssh/id_rsa` 암호 입력을 매번 하지 않기 위해 키체인에 등록하자
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
  * [tmux](/tmux/)
  * terminal multiplexer
* `nslookup`
  * `apt-get install dnsutils`
  * [nslookup 명령어 사용법 및 예제 정리](https://www.lesstif.com/pages/viewpage.action?pageId=20775988)
  * domain 을 주고 ip 로 확인하자.
  * `nslookup www.google.co.kr`
  * `nslookup -query=mx google.com`
    * DNS 중 MX RECORD 조회
  * `nslookup -q=cname example.com`
    * DNS 중 CNAME 조회
  * `nslookup -type=ns google.com`
    * NS RECORD 로 DNS 확인
  * `nslookup 209.132.183.181`
    * ip 를 주고 domain 을 찾는다.
  * `nslookup redhat.com 8.8.8.8`
    * name server 를 `8.8.8.8` 로 설정한다.
* `dig`
  * `apt-get install dnsutils`
  * DNS name server 와 도메인 설정이 완료된 후 DNS 질의 응답이 정상적으로 이루어지는 지를 확인한다.
  * `dig [@name server] [name] [query type]` 와 같은 형식으로 사용한다. 
    * `server` : name server 를 의미한다. 지정되지 않으면 `/etc/resolve.conf` 를 사용한다.
    * `query type`
      * `a` : network address
      * `any` : all query
      * `mx` : mail exchanger
      * `soa` : zone file 의 SOA 정보
      * `hinfo` : host info
      * `axfr` : zone transfer
      * `txt` : txt 값
  * `dig @kns1.kornet.net rootman.co.kr +trace`
    * kornet 에 rootman.co.kr 에 대한 계층적 위임 관계 요청
  * `dig . ns`
    * root name server 조회
  * `dig kr. ns`
    * 국내 kr name server 조회
  * `dig @kns1.kornet.net google.co.kr axfr`
    * zone transfer 이용 시, serial number 이후에 변경된 사항을 질의
  * `dig @kns.kornet.net txt chaos version.bind`
    * bind 버전 알아내기
  * `dig -x 216.58.200.3 +trace`
    * ip 를 주고 domain 을 확인하자.
* `curl`
  * [curl 설치 및 사용법 - HTTP GET/POST, REST API 연계등](https://www.lesstif.com/pages/viewpage.action?pageId=14745703)
  * URL 을 활용하여 data 전송하는 program. HTTP, HTTPS, RTMP 등등을 지원한다.
  * `curl "http://a.b.com/a?a=1"`
    * HTTP GET
  * `curl -X POST http://a.b.com`
    * `-X POST` 를 사용하여 HTTP POST 를 전송한다. `-d` 를 사용하면 `-X POST` 는 사용하지 않아도 HTTP POST 를 전송한다.
  * `curl --data "a=1&b=%20OK%20" http://a.b.com/a`
    * HTTP POST 를 데이터와 함께 전송하라.
  * `curl --data-urlencode "a=I am david" http://a.b.com/a`
    * HTTP POST 를 인코딩된 데이터와 함께 전송하라. 데이터에 공백이 있는 경우 유용하다.
  * `curl -i https://api.github.com/users/iamslash/orgs`
    * `-i` 를 사용하여 response 에 HTTP HEAD 를 포함하라. 
  * `curl -i https://api.github.com/users/iamslash/orgs -u <user-name>:<pass-word>`
    * authenticate with user-name, pass-word
  * `curl -H "Authorization: token OAUTH-TOKEN" https://api.github.com`
    * authenticate with OAuth2 token
`curl -d @a.js -H "Content-Type: application/json" --user-agent "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.14 (KHTML, like Gecko) Chrome/24.0.1292.0 Safari/537.14" http://a.b.com/a`
  * `-d` 를 이용하여 `a.js` 를 읽어서 HTTP POST 데이터로 전송한다.
  * `-H` 를 이용하여 HTTP HEAD 를 설정한다. 여러개의 HEAD 를 전송하고 싶다면 `-H` 를 여러개 사용하라.
  * `--user-agent` 를 이용하여 BROWSER 를 설정한다.
* `wget`
  * web 에서 파일좀 내려받아다오
  * `wget ftp://a.b.com/a.msg`
* `traceroute`
  * 네트워크 호스트까지 경로를 추적하자. 특정 라우터 까지 어떤 라우터들을 거쳐 가는가?
  * `traceroute www.google.com`
* `locate, updatedb`
  * 파일이름에 해당하는 걸 찾아다오
* `sudo`
  * 다른 유저로 command 를 실행하자.
  * `sudo find / -name "aaa"`
* `su`
  * EUID, EGID 를 수정하여 SHELL 을 실행하자.
  * `su - root`
* `bc`
  * 계산기 언어
  * `echo "56.8 + 77.7" | bc` 
  * `echo "scale=6; 60/7.02" | bc`
* `reset`
  * 터미널을 초기화 한다. 지저분 해졌을때 사용하면 좋음
* `tee`
  * stdin 으로 입력 받고 stdout 과 파일로 출력하자.
  * `ls | tee a.txt`
    * `ls > file` 은 stdout 말고 파일로만 출력한다.
  * `ls | tee -a a.txt`
    * stdout 으로 출력하고 파일레 추가하라.
  * `ls | tee a.txt b.txt c.txt`
* `script`
  * 갈무리
  * `script a.txt` `exit`
* `cut`
  * extract fields from a file
  ```bash
  $ cat a.txt
  aaa:bbb:Ccc:ddd:gdef
  efef:aab:wef:bgb:azc
  # -c : character position to cut
  $ cat a.txt | cut -c 3
  a
  e    
  $ cat a.txt | cut -c 1-5
  aaa:b
  efef:
  # -d : set field separater (default is TAB)
  # -f : field odd to cut
  $ cat a.txt | cut -d":" -f3
  Ccc
  wef
  $ cat a.txt | cut -d"K" -f3
  aaa:bbb:Ccc:ddd:gdef
  efef:aab:wef:bgb:azc
  # -s : If there is no field separator just skip
  $ cat a.txt | cut -d"K" -f3 -s
  ```
* `stat`
  * tell me the status of the file
  * `stat -c%s a.txt`
* `numfmt`
  * convert number format human readable
  * `numfmt --to=si --suffix=B --format="%.3f" $( stat -c%s a.txt )`
    * tell me the size of the file 'a.txt' such as 12B, 12KB, 12MB, 12GM.

## Process management

* `cpulimit`
  * [cpulimit @ github](https://github.com/opsengine/cpulimit)
  * throttle the cpu usage
  * `cpulimit -l 50 tar czvf a.tar.gz Hello`
    * tar with cpu usage under 50%.

* `nice`
  * handle process priorities (-20~19)
  * replace the current process image with the new process image which has the specific priority.
    ```bash
    $ ps -o pid,ni,comm
    # tar with nice
    $ nice -n 19 tar czvf a.tar.gz Hello
    ```
* `renice`
  * replace the specific process image with the new process image which has the specific priority.
    ```bash
    $ ps -o pid,ni,comm
    $ renice 19 12345 
    # change the priority of the processes with PIDs 987 and 32, plus all processes owned by the users daemon and root
    $ renice +1 987 -u daemon root -p 32
    ```
* `ionice`
  * handle process io priorities (0~7)
    ```bash
    # tar with ionice
    $ ionice -c 3 tar czvf a.tar.gz Hello
    # Sets process with PID 89 as an idle I/O process.
    $ ionice -c 3 -p 89
    # Runs 'bash' as a best-effort program with highest priority.
    $ ionice -c 2 -n 0 bash
    # Prints the class and priority of the processes with PID 89 and 91.    
    $ ionice -p 89 91
    ``` 

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

* [Linux Performance Analysis in 60,000 Milliseconds](https://medium.com/netflix-techblog/linux-performance-analysis-in-60-000-milliseconds-accc10403c55)
  * `uptime`
  * `dmesg | tail`
  * `vmstat -S M 1`
  * `mpstat -P ALL 1`
  * `pidstat 1`
  * `iostat -xz 1`
  * `free -m`
  * `sar -n DEV 1`
  * `sar -n TCP,ETCP 1`
  * `top`
  * `cat /proc/meminfo`
  
* `uptime`
  * `13:24:20 up  3:18,  0 users,  load average: 0.00, 0.01, 0.00`
  * 시스템이 `13:24:20` 부터 `3:18` 동안 살아있어
  * 1 분, 5 분, 15 분의 평균 load 를 보여줘
  * load 는 process 들 중 run, block 인 것들의 숫자야
  * `1 분 avg load < 5 분 avg load < 15 분 avg load` 이면 점점 load 가 늘어가는 추세이기 때문에 무언가 문제가 있다고 생각할 수 있다.
* `dmesg`
  * 커널의 메시지 버퍼를 보여다오
  * `dmesg | tail`
    * 마지막 커널의 메시지 버퍼 10 개를 보여다오
    * 치명적인 내용이 있는지 반드시 체크해야함
* `vmstat`
  * [vmstat에 대한 고찰(성능) 1편](http://egloos.zum.com/sword33/v/5976684)
  * [Vmstat에 대한 고찰(성능) 2편](http://egloos.zum.com/sword33/v/5997876)
  * [vmstat(8) - Linux man page](https://linux.die.net/man/8/vmstat)
  * virtual memory 통계 보여조

    | 범주     | 필드 이름  | 설명                                                                  |
    | ------ | ------ | ------------------------------------------------------------------- |
    | procs  | r      | The number of processes waiting for run time                        |
    |        | b      | The number of processes in uninterruptible sleep                    |
    | memory | swpd   | the amount of virtual memory used in KB                             |
    |        | free   | the amout of idle memory in KB                                      |
    |        | buff   | the amout of memory used as buffers in KB                           |
    |        | cache  | the amout of memory used as cache in KB                             |
    |        | inact  | the amout of inactive memory in KB                                  |
    |        | active | the amout of active memory in KB                                    |
    | swap   | si     | amount of memory swapped in from disk (/s)                          |
    |        | so     | amount of memory swapped to disk (/s)                               |
    | IO     | bi     | blocks received from a block device (blocks/s)                      |
    |        | bo     | amount of memory swapped to disk (blocks/s)                         |
    | system | in     | The number of interrupts per second. including the clock.           |
    |        | cs     | The number of context switches per second.                          |
    | CPU    | us     | Time spent running non-kernel code (user time, including nice time) |
    |        | sy     | Time spent running kernel code (system time)                        |
    |        | id     | Time spent idle, Prior to Linux 2.5.41, this inclues IO-wait time.  |
    |        | wa     | Time spent waiting for IO, Prior to Linux 2.5.41, inclues in idle.  |
    |        | st     | Time stolen from a virtual machine, Prior to Linux 2.5.41, unknown. |

  * `vmstat 1`
    * 1 초 마다 보여다오
    ```bash
    procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
    r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
    6  0      0 376096  93376 788776    0    0    15    34  234   67  2  3 94  1  0
    ```
  * `vmstat -S M 1`
    * 1 초 마다 MB 단위로 보여다오
  * `total physical memory = free + buff + cache + used`
    * buff 는 i-node 값 즉 파일들의 실제 주소를 보관한다. disk seek time 을 최소화 할 수 있다.
    * cache 는 파일의 real data 를 cache 한다. disk read performance 를 향상시킬 수 있다.
    * free 가 부족하면 cache 에서 free 로 메모리가 옮겨갈 수도 있다. free 가 부족하다고 꼭 메모리가 부족한 상황은 아니다.
    * `/proc/sys/vm/vfs_cache_pressure` 가 buff 와 cache 의 비율을 설정하는 값이다. default 는 100 이다. 파일의 개수가 많아서 buff 가 중요한 시스템이라면 이 것을 높게 설정한다.
    * `/proc/sys/vm/min_free_kbytes` 는 free 의 최소 용량이다. 이것이 낮아지면 cache 가 높아진다. hit ratio 가 낮은 시스템인 경우 cache 가 필요 없으므로 min_free_kbytes 를 늘려주자.
    * `/proc/sys/vm/swappiness` 는 swapping 하는 정도이다. 이것이 높으면 cache 를 삭제하는 것보다 swapping 하는 비율이 높아진다. 이 것이 낮으면 swapping 하는 것보다 cache 를 삭제하는 비율이 높아진다. 이 것을 0 으로 설정하면 swapping 을 하지 않기 때문에 disk 와 memory 사이에 데이터 전송이 발생하지 않는다. memory 가 낮고 memory 사용량이 높은 시스템의 경우 swappiness 를 0 보다 크게 설정해야 한다.
  * `r` 이 `CPU core` 보다 크면 CPU 의 모든 core 가 일을 하고 있는 상황이다.
  * `b` 가 `CPU core` 보다 크면 disk write bottle neck 일 수 있다.
  * `wa` 가 크면 disk read bottle neck 일 수 있다.
  * `si, so` 가 0 이 아니면 메모리가 부족한 상황이다.
  * `id` 가 작으면 CPU 가 바쁜 상황이다.
  * `in` 은 인터럽트이다. 주변장치에서 CPU 에 자원을 요청하는 횟수이다. 일반 적인 컴퓨터에서 마우스의 인터럽트가 많지만 서버의 경우는 이더넷장치와 DISK 의 인터럽트가 많다.
  * `swapd` 는 Virtual Memory 즉 DISK 에 상주하는 VM 의 크기이다.
  * active memory are pages which have been accessed "recently", inactive memory are pages which have not been accessed "recently"
    * [Linux inactive memory @ stackexchange](https://unix.stackexchange.com/questions/305606/linux-inactive-memory)
  * `vmstat -s` 부트이후 통계
    ```
            1999 M total memory
             769 M used memory
             782 M active memory
             713 M inactive memory
             367 M free memory
              91 M buffer memory
             770 M swap cache
            1023 M total swap
               0 M used swap
            1023 M free swap
             94387 non-nice user cpu ticks
                 0 nice user cpu ticks
            115807 system cpu ticks
           4524005 idle cpu ticks
             57978 IO-wait cpu ticks
                 0 IRQ cpu ticks
             11650 softirq cpu ticks
                 0 stolen cpu ticks
            704619 pages paged in
           1647336 pages paged out
                 0 pages swapped in
                 0 pages swapped out
          11245885 interrupts
          89013771 CPU context switches
        1573848951 boot time
              3959 forks
    ```

* `mpstat`
  * `apt-get install sysstat`
  * CPU 상황을 자세히 모니터링한다.

    | name    | desc                                                                                                                                  |
    | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
    | CPU     | Processor number                                                                                                                      |
    | %usr    | while executing at the user level (application)                                                                                       |
    | %nice   | while executing at the user level with nice priority.                                                                                 |
    | %sys    | while executing at the system level (kernel)                                                                                          |
    | %iowait | percentage of time that the CPU or CPUs were idle during which the system had an outstanding disk I/O request.                        |
    | %irq    | percentage of time spent by the CPU or CPUs to service hardware interrupts.                                                           |
    | %soft   | percentage of time spent by the CPU or CPUs to service software interrupts.                                                           |
    | %steal  | percentage of time spent in involuntary wait by the virtual CPU or CPUs while the hypervisor was servicing another virtual processor. |
    | %guest  | percentage of time spent by the CPU or CPUs to run a virtual processor.                                                               |
    | %gnice  | percentage of time spent by the CPU or CPUs to run a niced guest.                                                                     |
    | %idle   | percentage of time that the CPU or CPUs were idle and the system did not have an outstanding disk I/O request.                        |
  * `mpstat -P ALL 1`
    * 1 초 마다 모든 CPU 에 대해 보여줘

* `pidstat`
  * process 별로 CPU 의 점유율을 확인할 수 있다.
  * `pidstat`
    ```
    Linux 4.9.184-linuxkit (86e6c5bfb041)   11/16/19        _x86_64_        (2 CPU)

    03:59:53      UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command
    03:59:53        0         1    0.0%    0.0%    0.0%    0.0%    0.0%     1  bash
    03:59:53        0        10    0.0%    0.0%    0.0%    0.0%    0.0%     1  bash    
    ```
  * `pidstat 1`
    * 1 초 마다 보여도

* `iostat`
  * block device 별로 io 를 모니터링 하자.
  * `man iostat`
  * `iostat`
    ```
    Linux 4.9.184-linuxkit (86e6c5bfb041)   11/16/19        _x86_64_        (2 CPU)

    avg-cpu:  %user   %nice %system %iowait  %steal   %idle
              2.10    0.00    2.81    1.20    0.00   93.88

    Device             tps    kB_read/s    kB_wrtn/s    kB_dscd/s    kB_read    kB_wrtn    kB_dscd
    sda              18.40        14.87        78.69         0.00     416883    2205424          0
    sr0               0.09         6.74         0.00         0.00     188798          0          0
    sr1               0.00         0.02         0.00         0.00        470          0          0
    sr2               0.05         3.52         0.00         0.00      98584          0          0    
    ``` 
  * `iostat 1`
    * 1 초마다 보여줘
  * `iostat -xz 1`
    * `r/s, w/s, rkB/s, wkB/s` 는 각각 초당 읽기, 쓰기, kB읽기, kB 쓰기를 의미한다.
    * `await` : The average time for the I/O in milliseconds.
    * `avgqu-sz` : The average number of requests issued to the device. 
    * `%util` : Device utilization. 
* `free`
  * physical memory 와 swap memory 의 상태를 알려다오
    * `total physical memory = used + free + shared + buffers + cached`
    * `buffers` : For the buffer cache, used for block device I/O, saves i-node data to reduce DISK seek time.
    * `cached` : For the page cache, used by file systems, saves file data to reduce I/O.
    * `available` : include `free` and a part of `buff/cache`.
  * `free -h`
    * human readable 하게 보여줘
  * `free -ht` total 추가해조
  * `free -hts 5` 5초마다 갱신해서 보여줘

* `sar`
  * network interface throughput
  * `Cannot open /var/log/sa/sa16: No such file or directory` error 해결 방법
    ```
    $ vim /etc/default/sysstat
    ENABLED = true
    ```
  * `sar -n DEV 1`
  * `sar -n TCP,ETCP 1`
  * `sar -u 1 3` 모든 CPU를 1초마다 갱신해서 3번 보여조
  * `sar -r 1 3` 메모리 보여줘
  * `sar -S` 스왑공간 보여줘
  * `sar -b 1 3` I/O 활동 보여줘
  * `sar -d 1 1` block device 별로 보여줘
  * `sar -p -d 1 1` 예쁘게 보여줘 pretty print
  * `sar -w 1 3` context switch 보여줘
  * `sar -q 1 3` load average 보여줘
  * `sar -n DEV 1 1` eth0, eth1보여줘
  * `sar -q -f /var/log/sa/sa23 -s 10:00:01` 10:00:01부터 통계만들어서 보여줘
  * `sar -q -f /var/log/sa/sa23 -s 10:00:01 | head -n 10` 10:00:01부터 통계 만들어서 10개만 보여줘
* `top` `htop` `atop`
  * `top -n 10` 10번만 갱신해
  * `top -n 1 -b > a.txt` export
  * `top -u iamslash` 특정 유저 소유의 프로세스들만 보여줘
  * `z` running 프로세스들은 색깔표시해조
  * `c` 절대 경로 
  * `d` 갱신 시간 조정
  * `k` 프로세스에게 SIGKILL전송
  * `r` 프로세스의 nice를 변경하여 우선순위를 조정한다.
  * `SHIFT + m` 메모리 사용량이 큰 순서대로 정렬
  * `SHIFT + p` cpu 사용량이 큰 순서대로 정렬
  * `SHIFT + t` 실행시간이 큰 순서대로 정렬

* meminfo
  * [리눅스 메모리 정보](https://classpath.tistory.com/306)
  * [[Linux] Cached Memory 비우고 Free Memory 늘리기](http://egloos.zum.com/mcchae/v/11217429)
  * [[Memory] 커널 메모리 관리](https://fmyson.tistory.com/212)

    | item         | desc                  |
    | ------------ | --------------------- |
    | MemTotal     | total physical memory |
    | MemFree      |                       |
    | Buffers      |                       |
    | Cached       |                       |
    | SwapCache    |                       |
    | Active       |                       |
    | Inactive     |                       |
    | HighTotal    |                       |
    | LowTotal     |                       |
    | LowFree      |                       |
    | SwapTotal    |                       |
    | SwapFree     |                       |
    | Dirty        |                       |
    | Writeback    |                       |
    | Committed_AS |                       |
    | Slab         |                       |

  * `cat /proc/meminfo`
    ```    
    MemTotal:        2047016 kB
    MemFree:          371632 kB
    MemAvailable:    1100604 kB
    Buffers:           96416 kB
    Cached:           741020 kB
    SwapCached:            0 kB
    Active:           804832 kB
    Inactive:         731148 kB
    Active(anon):     554092 kB
    Inactive(anon):   145844 kB
    Active(file):     250740 kB
    Inactive(file):   585304 kB
    Unevictable:           0 kB
    Mlocked:               0 kB
    SwapTotal:       1048572 kB
    SwapFree:        1048572 kB
    Dirty:                 8 kB
    Writeback:             0 kB
    AnonPages:        695892 kB
    Mapped:           282732 kB
    Shmem:              1396 kB
    Slab:              71876 kB
    SReclaimable:      49464 kB
    SUnreclaim:        22412 kB
    KernelStack:        6688 kB
    PageTables:         3672 kB
    NFS_Unstable:          0 kB
    Bounce:                0 kB
    WritebackTmp:          0 kB
    CommitLimit:     2072080 kB
    Committed_AS:    3489592 kB
    VmallocTotal:   34359738367 kB
    VmallocUsed:           0 kB
    VmallocChunk:          0 kB
    AnonHugePages:     10240 kB
    ShmemHugePages:        0 kB
    ShmemPmdMapped:        0 kB
    HugePages_Total:       0
    HugePages_Free:        0
    HugePages_Rsvd:        0
    HugePages_Surp:        0
    Hugepagesize:       2048 kB
    DirectMap4k:       32052 kB
    DirectMap2M:     2064384 kB
    DirectMap1G:           0 kB    
    ```

* `du`
  * `du -h /home/iamslash` human readable format
  * `du -sh /home/iamslash` 요약해서 알려줘
  * `du -h -d 1 /home/iamslash` depth 1 directory 까지 알려줘
  * `du -ah /home/iamslash` 모든 파일과 디렉토리 알려줘
  * `du -hk /home/iamslash` KB
  * `du -hm /home/iamslash` MB
  * `du -hc /home/iamslash` 마지막에 total 보여줘
  * `du -ah --exclude="*.txt" /home/iamslash`
  * `du -ah --time /home/iamslash` last modified time 도 보여줘
* `df`
  * `df -h` human readable format
  * `df -ah` 모두 보여다오
  * `df -ih` i-node 정보좀 털어봐라
  * `df -Th` 파일 시스템 타입좀
  * `df -th ext3` ext3 포함해서 알려줘
  * `df -xh ext3` ext3 빼고 알려줘
* `netstat`
  * `apt install net-tools`
  * 네트워크 상태좀 알려다오
  * `netstat -lntp` tcp 로 리스닝하는 프로세스들 보여줘
  * `netstat -a | more` TCP, UDP 포트 모두 보여줘
  * `netstat -at` TCP 포트만 보여줘
  * `netstat -au` UDP 포트만 보여줘
  * `netstat -l` listening 포트 보여줘
  * `netstat -lt` TCP listening 포트 보여줘
  * `netstat -lu` UDP listening 포트 보여줘
  * `netstat -lx` UNIX domain socket listening 포트 보여줘
  * `netstat -s` 프로토콜(TCP, UDP, ICMP, IP)별로 통계를 보여줘
  * `netstat -st` `netstat -su`
  * `netstat -tp` TCP 사용하는 녀석들을 PID/programname 형태로 보여줘
  * `netstat -ac 5 | grep tcp` 5초마다 갱신하면서 promiscuous mode인 녀석들 보여줘
  * `netstat -r` routing table보여줘
  * `netstat -i` network interface별로 MTU등등을 보여줘
  * `netstat -ie` kernal interface table 보여줘 ifconfig와 유사
  * `netstat -g` multicast group membership information을 보여줘
  * `netstat -c` 몇초마다 갱신하면서 보여줘
  * `netstat --verbose`
  * `netstat -ap | grep http`
  * `netstat --statistics --raw` 아주 심한 통계를 보여달라
* `ss`
  * socket statistics. netstat 과 옵션의 의미가 유사하다.
  * `ss -plat` tcp 로 리스닝하는 프로세스들 보여줘
  * `ss | less` 모든 연결을 보여다오
  * `ss -t` TCP `ss-u` UDP `ss-x` UNIX
  * `ss -nt` hostname얻어 오지 말고 숫자로만 보여줘
  * `ss -ltn` TCP listening socket보여줘
  * `ss -ltp` TCP listening socket들을 PID/name와 함께 보여줘
  * `ss -s` 통계 보여줘
  * `ss -tn -o` timer 정보도 함께 보여줘
  * `ss -tl -f inet` `ss -tl -4` IPv4 연결만 보여줘
  * `ss -tl -f inet6` `ss -tl -6` IPv6 연결만 보여줘
  * `ss -t4 state established` `ss -t4 state time-wait`
  * `ss -at '( dport = :ssh or sport = :ssh )'` source, destination 이 ssh인 것만 보여줘
  * `ss -nt '( dst :443 or dst :80 )'`
  * `ss -nt dst 74.125.236.178` `ss -nt dst 74.125.236.178/16` `ss -nt dst 74.125.236.178:80`
  * `ss -nt dport = :80` `ss -nt src 127.0.0.1 sport gt :5000`
  * `sudo ss -ntlp sport eq :smtp` `sudo ss -nt sport gt :1024`
  * `sudo ss -nt dport \< :100` `sudo ss -nt state connected dport = :80`

* `sysstat` `sar` `sadc`
  * 시스템 통계를 위한 툴들의 모임이다.
    * **sar** collects and displays ALL system activities statistics.
    * **sadc** stands for “system activity data collector”. This is the sar backend tool that does the data collection.
    * **sa1** stores system activities in binary data file. sa1 depends on sadc for this purpose. sa1 runs from cron.
    * **sa2** creates daily summary of the collected statistics. sa2 runs from cron.
    * **sadf** can generate sar report in CSV, XML, and various other formats. Use this to integrate sar data with other tools.
    * **iostat** generates CPU, I/O statistics
    * **mpstat** displays CPU statistics.
    * **pidstat** reports statistics based on the process id (PID)
    * **nfsiostat** displays NFS I/O statistics.
    * **cifsiostat** generates CIFS statistics.
  * `sudo vi /etc/default/sysstat` `ENABLED="true"` 매 10분마다 데이터 수집을 위한 sadc 활성화
    * `sudo vi /etc/cron.d/sysstat` `* * * * * root command -v debian-sa1 > /dev/null && debian-sa1 1 1` 매분마다 해볼까

* `ifconfig`
  * network interface parameter설정하기
  * `ifconfig eth0`
  * `ifconfig -a` disable된 network interface까지 몽땅 보여다오
  * `ifconfig eth0 down` `ifconfig eth0 up`
  * `ifconfig eth0 192.168.2.2` eth0 에 ip할당
  * `ifconfig eth0 netmask 255.255.255.0` eth0에 subnet mask 할당
  * `ifconfig eth0 broadcast 192.168.2.255` eth0의 broadcast address교체
  * `ifconfig eth0 192.168.2.2 netmask 255.255.255.0 broadcast 192.168.2.255`
  * `ifconfig eth0 mtu XX` eth0의 maximum transmission unit교체. 기본값은 1500
  * `ifconfig eth0 promisc`
* `lsof`
  * 열린 파일들을 보여도
  * `lsof /var/log/syslog`
  * `lsof +D /var/log/` 특정 디렉토리 이하를 재귀적으로 보여다오 
  * `lsof +d /var/log/` 특정 디렉토리만 보여다오
  * `lsof -c ssh -c init` ssh 혹은 init으로 시작하는 command들만 보여다오 
  * `lsof -u iamslash` iamslash 유저만 보여다오
  * `lsof -u ^iamslash`
  * `lsof -p 1753` PID가 1753인 것만 보여다오
  * `lsof -t -u iamslash` user가 iamslash인 녀석들의 PID들을 보여다오
  * `kill -9 'lsof -t -u iamslash'` user가 iamslash인 녀석들에게 SIGKILL을 보내다오 
  * `lsof -u iamslash -c ssh` `lsof -u iamslash -c ssh -a` -a는 ssh시작하는을 의미
  * `lsof -u iamslash -c ssh -a -r5` 5초마다 갱신해서 보여다오
  * `lsof -i` 접속된 연결을 보여다오
  * `lsof -i -a -c ssh`
  * `lsof -i :25` 25번 포트에 접속한 연결을 보여다오
  * `lsof -i tcp` `lsof -i udp`
  * `lsof -N -u iamslash -a` NFS보여다오
* `lshw`
  * 하드웨어 정보들을 알려다오
  * `sudo lshw`
  * `sudo lshw -short` 짧게 부탁해
  * `sudo lshw -short -class memory` 메모리분야만 부탁해
  * `sudo lshw -class processor` `sudo lshw -short -class disk` `sudo lshw -class network`
  * `sudo lshw -businfo` pci, usb, scsi, ide자치들의 주소를 부탁해
  * `sudo lshw -html > hardware.html` `sudo lshw -xml > hardware.xml`
* `who`
  * 로그인한 녀석들 보여다오
* `date`
  * 날짜 시간을 datetime 형식으로 출력해다오
  * `date +"%Y-%m-%d %H:%M:%S"`
  * `date --date="12/2/2014"`
  * `date --date="next mon"`
  * `date --date=@5` UTC 이후로 5초 지났다.
  * `date --date='3 seconds ago'`
* `time`
  * command 실행하고 소요시간 출력해다오
  * `time a.out`
* `uname`
  * 운영체제이름 알려다오
  * `uname -a`
* `hostname`
  * 호스트이름 알려다오
* `last`
  * 마지막에 로그인했던 user, tty, host알려다오
* `ping`
  * ICMP ECHO_REQUEST packet를 보낸다.
  * `ping -i 5 127.0.0.1` 패킷 보내기 전에 5초 쉬어
  * `ping 0` `ping localhost` `ping 127.0.0.1` localhost살아있니?
  * `ping -c 5 www.google.com` 패킷 5개만 보내자
  * `ping -f localhost` 몇초만에 400,000개 이상의 패킷들을 보낸다.
  * `ping -a www.google.com` peer가 살아 있다면 소리 발생
  * `ping -c 1 google.com` 도메인 이름으로 아이피좀 알자
  * `ping -c 5 -q 127.0.0.1` ping summary
  * `ping -s 100 localhost` packet size를 56에서 100으로 변경
  * `ping -w 5 localhost` 5초동안만 보내자.

## 로그

* `grep`
  * 파일 패턴 검색기
  * `grep "this" a.txt`
  * `grep "this" a.*`
  * `grep -l this a.*`
    * files with matches recursively
  * `grep -r "this" *`
  * `grep -v "go" a.txt` 검색 대상이 안된 줄을 보여줘라
  * `grep -c "go" a.txt` 검색된 수를 보여다오
  * `grep -i "the" a.txt` case insensitive
  * `grep "lines.*empty" a.txt` regular expression
  * `grep -i "is" a.txt`
  * `grep -iw "is" a.txt` 검색의 대상은 단어이다.
  * `grep -A 3 -i "example" a.txt` 검색된 줄 이후 3줄 더 보여줘라
  * `grep -B 2 "single WORD" a.txt` 검색된 줄 이전 2줄 더 보여줘라
  * `grep -C 2 "example" a.txt` 검색된 줄 이전 이후 2줄 더 보여줘라
  * `export GREP_OPTIONS='--color=auto' GREP_COLOR='100;8'` `grep this a.txt`
  * `grep -o "is.*line" a.txt` 검색된 문자열들만 보여다오
  * `grep -o -b "3" a.txt` 검색된 위치를 보여다오
  * `grep -n "go" a.txt` 검색된 줄번호도 보여다오
* `grep` vs `egrep` vs `fgrep`
  * `grep` is `grep -G`
  * `egrep` is `grep -E` or `grep --extended-regexp`
    * 확장 정규표현식을 사용할 때 메타캐릭터를 escape 하지 않아도 된다.
    * `egrep 'no(foo|bar)' a.txt` 와 `grep 'no\(foo\|bar\)' a.txt` 은 결과가 같다.
  * `fgrep` is `grep -F` or `grep --fixed-strings`
    * 정규표현식을 사용하지 않겠다는 의미이다.
    * `fgrep 'foo.' a.txt`
      * `foo.` 만 검색된다.
    * `grep 'foo.' a.txt`
      * `fooa, foob, fooc` 등이 검색된다.
* `xargs`
  * 구분자 `\n` 을 이용해서 argument list 를 구성하고 command 한개에 argument 하나씩 대응해서 실행하자
  * xargs 가 실행할 command 가 없다면 `/bin/echo`를 사용한다.
  * `echo a b c d e f | xargs`
  * `echo a b c d e f | xargs -n 3` argument는 3개씩 한조가 되라
  * `echo a b c d e f | xargs -p -n 3` prompt 등장
  * `ls -t | xargs -I % sh -c "echo %; echo %"`
    * % 를 argument 로 해서 여러 command 를 실행하라
  * `find . -name "*.c" | xargs rm -rf`
  * `find . -name "*.c" -print0 | xargs -0 rm -rf`
  * `find . -name '*.c' | xargs grep 'stdlib.h'`
* `find`
  * `find . -name "*.log" | xargs cat`
  * `find . -name a.txt`
  * `find /home -name a.txt`
  * `find /home -iname a.txt` case insensitive
  * `find / -type d -name a`
  * `find . -type f -name a.php`
  * `find . -type f -name "*.php"`
  * `find . -type f -perm 0777 -print`
  * `find / -type f ! -perm 777` without permission 777
  * `find / -perm 2644` sgid bit with 644
  * `find / -perm 1551` stcky bit with 551
  * `find / -perm /u=s` suid
  * `find / -perm /g=s` sguid
  * `find / -perm /u=r` read only
  * `find / -perm /a=x` executable
  * `find / -type f -perm 0777 -print -exec chmod 644 {} \;`
  * `find / -type d -perm 777 -print -exec chmod 755 {} \;`
  * `find . -type f -name "a.txt" -exec rm -f {} \;`
  * `find . -type f -name "*.txt" -exec rm -f {} \;`
  * `find /tmp -type f -empty`
  * `find /tmp -type d -empty`
  * `find /tmp -type f -name ".*"`
  * `find / -user root -name a.txt`
  * `find /home -user iamslash`
  * `find /home -group staff`
  * `find /home -user iamslash -name "*.txt"`
  * `find / -mtime 50` 수정시간이 50일이내인 것
  * `find / -mtime +50 -mtime -100` 수정시간이 50일 이상 100일 이하인 것
  * `find / -cmin -60` 작성 시간이 60분 이내인 것
  * `find / -mmin -60` 수정 시간이 60분 이내인 것
  * `find / -amin -60` 접근 시간이 60분 이내인 것
  * `find / -size 50M`
  * `find / -size +50M -size -100M`
  * `find / -size +100M -exec rm -rf {} \;`
  * `find / -type f -name *.mp3 -size +10M -exec rm {} \;`
* `tail`
  * `tail -n 5 a.txt`
    * a.txt 의 끝에서 5 줄을 출력하라.
  * `tail -n +5 a.txt`
    * a.txt 의 5 줄부터 출력하라. 
  * `tail -f /var/log/messages`
  * `tail -f /tmp/a.log --pid=2575` 2575프로세스 죽을때까지
  * `tail -f /tmp/debug.log --retry`
* `head`
  * `head -n 5 a.txt`
  * `head -5 a.txt`
  * `ls | head -5`
* `awk`
  * [awk](/awk/README.md)
  * `cat a.txt | awk '{print $4}'`
    * a.txt 를 읽어서 매 줄마다 4 번 째 컬럼만 출력하라
* `uniq`
  * `uniq a.txt` 유일한 라인만 출력
  * `uniq -c a.txt` 중복된 라인의 개수를 출력
  * `uniq -d a.txt` 중복된 라인들을 하나씩만 출력.
  * `uniq -D a.txt` 중복된 라인들을 모두 출력
  * `uniq -u a.txt` 유일한 라인들을 출력
  * `uniq -c -w 8 a.txt` 처음 8 바이트만 중복되지 않은 라인들
  * `uniq -D -w 8 a.txt` 처음 8 바이트만 중복된 라인들 모두 보여다오
  * `uniq -D -s 2 a.txt` 2 바이트는 건너띄고 중복된 라인들 모두
  * `uniq -D -f 2 a.txt` 2 필드는 건너띄고 중복된 라인들 모두
* `sort`
  * `sort a.txt > b.txt`
  * `sort -r a.txt > b.txt`
    * sort reversly
  * `sort -nk2 a.txt` 2열을 기준으로 정렬해라.
  * `sort -k9 a.txt`
  * `ls -l /home/$USER | sort -nk5`
  * `sort -u a.txt` 정렬하고 중복된 거 지워라
  * `sort a.txt b.txt`
  * `sort -u a.txt b.txt`
  * `ls -l /home/$USER | sort -t "," -nk2,5 -k9` 숫자인 2, 5 열, 문자인 9 열을 기준으로 정렬해라.
* `wc`
  * `wc a.txt` 줄개수, 단어수, 바이트 표시해조
  * `wc -l a.txt` 줄개수 보여줘
  * `wc -w a.txt` 단어수 보여줘
  * `wc -c a.txt` 바이트수 보여도
  * `wc -m a.txt` 바이트수 보여도
  * `wc -L a.txt` 가장 길이가 긴 행의 문자개수

## 텍스트

* `paste`
  * 파일의 특정 행들을 머지해줘
  * `seq 1 10 | paste - - -`
    * `seq 1 10 | column`
  * `paste a.txt`
  * `paste -s a.txt` 모든 행을 join 해줘
  * `paste -d, -s a.txt` comma 를 구분자로 모든 행을 join 해도
  * `paste - - < a.txt` 2열로 merge해라.
  * `paste -d':' - - < a.txt` 구분자는 `:` 로 하고 2 열로 merge 해라.
  * `paste -d ':,' - - - < a.txt` 구분자는 `:` `,` 로 하고 3 열로 merge 해라.
  * `paste a.txt b.txt` 파일 두개 2 열로 merge 하라.
  * `paste -d, a.txt b.txt` 구분자는 `,` 으로 파일 두개 2 열로 merge 해라.
  * `cat b.txt | paste -d, a.txt -`
  * `cat a.txt | paste -d, - b.txt`
  * `cat a.txt b.txt | paste -d, - -`
  * `paste -d'\n' a.txt b.txt`
* [sed](/sed/)
  * `sed -e 's/regex/REGEXP/g' a.txt`
    * a.txt 를 읽어서 모든 regex 를 REGEXP 로 바꿔라
  * `sed -e 's/regex/REGEXP/g' a.txt > b.txt`
  * `sed -e 's/regex/REGEXP/g' -i a.txt`
    * a.txt 를 읽어서 모든 regex 를 REGEXP 로 바꿔서 a.txt 에 저장하라
* `jq`
  * [커맨드라인 JSON 프로세서 jq @ 44bits](https://www.44bits.io/ko/post/cli_json_processor_jq_basic_syntax)
  * sed for json
    ```bash
    $ jq '.'
    {"foo": "bar"}
    {
      "foo": "bar"
    }

    $ echo 'null' | jq '.'
    null

    $ echo '"String"' | jq '.'
    "String"

    $ echo '44' | jq '.'
    44    

    $ echo '{"foo": "bar", "hoge": "piyo"}' | jq '.foo'
    "bar"   

    $ echo '{"a": {"b": {"c": "d"}}}' | jq '.a.b.c'
    "d"     

    $ echo '{"a": {"b": {"c": "d"}}}' | jq '.a | .b | .c'
    "d"    

    $ echo '[0, 11, 22, 33, 44 ,55]' | jq '.[4]'
    44 

    $ echo '{"data": [0, 11, 22, 33, 44 ,55]'} | jq '.data | .[4]'
    44       
    ```
  * 
* `tr`
  * 문자열 번역해줘
  * `echo HELLO | tr "[:upper:]" "[:lower:]"` 대문자를 소문자로 바꾸어 다오
  * `echo HELLO | tr "[A-Z]" "[a-z]"` 대문자를 소문자로 바꾸어 다오
  * `tr '{}' '()' < a.txt > b.txt`
  * `echo "the geek stuff" | tr -d 't'` t문자 지워줘
  * `echo "my username is 432234" | tr -d "[:digit:]"` 숫자 지워줘
  * `echo "my username is 432234" | tr -cd "[:digit:]` 숫자 빼고 지워줘
  * `tr -cd "[:print:]" < a.txt` non printable 문자들을 지워다오
  * `tr -s '\n' ' ' < a.txt` 모든 행을 join 하자.
* `printf`
  * format string을 출력해줘
  * `printf "%s\n" "hello printf"`
  * `printf "%s\n" "hllo printf" "in" "bash script"` 각 문자열에 적용해서 출력해라.
* `nl`
  * 줄번호 보여줘라
  * `nl a.txt`
  * `nl a.txt > nla.txt`
  * `nl -i5 a.txt` 줄번호는 5씩 더해서 보여다오
  * `nl -w1 a.txt` 줄번호 열위치를 다르게 해서 보여다오
  * `nl -bpA a.txt` A로 시작하는 문자열만 줄번호 적용해라
  * `nl -nln a.txt` `nl -nrn a.txt` `nl -nrz a.txt`
* `seq`
  * sequence number 출력해 다오
  * `seq 10`
  * `seq 35 45`
  * `seq -f "%02g/03/2016" 31`
  * `seq 10 -1 1` 거꾸로 보여줘 
* `sha1sum`
  * SHA-1 message digest알려줄래?
  * `sha1sum a.txt`
* `md5sum`

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
  * system call 과 signal 을 모니터링 해주라. 소스는 없는데 디버깅 하고 싶을때 유용하다
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
  * library call 좀 보여다오
  * `ltrace ls`
  * `ltrace -p 1275`
* `pstack`
  * process 의 callstack 을 thread 별로 보여다오
  * `pstack 1275` 
* `pmap`
  * process의 memory map좀 보여다오
  * `pmap 1275`
  * `pmap -x 1275` show details
  * `pmap -d 1275` show the device format
  * `pmap -q 1275` header 와 footer 는 보여주지 말아라
* `valgrind`
  * 메모리 누수를 검사하자.
  * `valgrind --leak-check=yes myprog arg1 arg2`
* `od`
  * octal numbers(8진수)로 보여다오
  * `od -b a.txt` 8 진수로 보여줘
  * `od -c a.txt` 캐릭터로 보여줄래?
  * `od -Ax -c a.txt` byte offset 을 hexadecimal 형식으로 보여다오
  * `od -An -c a.txt` byte offset 제외 하고 보여다오
  * `od -j9 -c a.txt` 9 bytes 건너뛰고 보여다오
  * `od -N9 -c a.txt` 9 bytes 만 보여다오
  * `od -i a.txt` decimal 형식으로 보여다오
  * `od -x a.txt` hexadecimal 2 byte단위로 보여다오
  * `od -o a.txt` octal 2 byte단위로 보여다오
  * `od -w1 -c -Ad a.txt`
  * `od -w1 -v -c -Ad a.txt` 중복된 것도 보여줘
* `strings`
  * 최소한 4 byte 보다 크거나 같은 문자열을 보여다오
  * `strings a.out`
  * `strings a.out | grep hello`
  * `strings -n 2 welcome | grep ls`
  * `strings -o a.out` offset도 알려줘
  * `strings -f /bin/* | grep Copy` 여러개의 파일들의 스트링을 알려줘

## 압축

* `tar`
  * `tar -cvf a.tar /home/iamslash/tmp`
  * `tar -czvf a.tar -C /home/iamslash/ tmp`
    * `/home/iamslash/` 에서 tmp 를 압축해라
  * `tar -czvf a.tar.gz /home/iamslash/tmp`
  * `tar -cjvf a.tar.gz2 /home/iamslash/tmp`
  * `tar -xvf a.tar`
  * `tar -xzvf a.tar.gz`
  * `tar -xjvf a.tar.gz2`
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
  * gzip, bzip2 보다 압축률이 더욱 좋다.
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
  * less with gz files
  * `zless a.txt.gz`
* `bzless`
  * less with bz2 files
  * `bzless a.txt.bz2`

## 에디터

* `vi, vim, emacs, nano, ed`

## 데몬 관리

* cron
  * 반복 예약 작업 등록
  * 시간의 형식은 `MIN HOUR DOM MONTH DOW` 와 같다.
    * `MIN` : minutes within the hour (0-59)
    * `HOUR` : the hour of the day (0-23)
    * `DOM` : The day of the month (1-31)
    * `MONTH` : The month (1-12)
    * `DOW` : The day of the week (0-7), where 0 and 7 are Sundays.
    * `*` specifies all valid values
    * `M-N` specifies a range of values
    * `M-N/X` or `*/X` steps by intervals of X through the specified range or whole valid range
    * `A,B,...,Z` enumerates multiple values
  * `crontab -l` 등록된 목록을 보여줘
  * `crontab -l -u iamslash` iamslash USER 의 목록 보여줘
  * `crontab -e` 크론탭 수정해볼까
    * `* * * * * /tmp/a.sh` 매 1분 마다 실행해 
    * `15,45 * * * * /tmp/a.sh` 매시 15, 45분 마다 실행해 
    * `*/10 * * * * /tmp/a.sh` 10 분 마다 실행해 
    * `0 2 * * * /tmp/a.sh` 매일 02:00 에 마다 실행해 
    * `30 */6 * * * /tmp/a.sh` 매 6 시간 마다(00:30, 06:30, 12:30, 18:30) 실행해 
    * `30 1-23/6 * * * /tmp/a.sh` 1 시부터 23 시까지 매 6시간 마다(01:30, 07:30, 13:30, 19:30) 실행해 
    * `0 8 * * 1-5 /tmp/a.sh` 평일(월-금) 08:00 
    * `0 8 * * 0,6 /tmp/a.sh` 주말(일,토) 08:00
  * `crontab -r` 모두 삭제
  * `* * * * * /tmp/a.sh > /dev/null 2>&1`
    * run `/tmp/a.sh` every min without logs
  * `* * * * * /tmp/a.sh > /tmp/a.txt`
    * run `/tmp/a.sh` every min and write STDOUT to `/tmp/a.txt`
  * `* * * * * /tmp/a.sh >> /var/log/a.log`
    * run `/tmp/a.sh` every min and append STDOUT to `/var/log/a.log`
  * cron logs
    * `$ cat /var/log/cron`
  * view crontab
    * `$ cat /var/spool/cron/crontab/root`

* systemd
  * `System has not been booted with systemd as init system (PID 1). Can't operate.` error 해결 방법???
  * `systemctl` 현재 작동하고 있는 서비스들 보여줘
  * `systemctl list-unit-files` 작동하지 않아도 좋으니 모든 서비스들 보여줘
  * `systemctl enable vsftpd` 리눅스 부팅할때 vsftpd 시작해줘
    * `disable` `start` `stop` `restart` `reload` 
  * `systemctl is-enabled vsftpd` `systemctl is-active vsftpd`
  * `systemctl show http.service` 서비스에 대한 systemd설정 보기
  * `systemctl list-dependencies` 각 서비스들의 의존관계
  * `systemctl list-dependencies mariadb.service` 특정 서비스의 의존관계
  * `systemctl daemon-reload` systemd를 다시 시작
  * unit file 편집하기
    * `/usr/lib/system/system/ntpd.service`를 편집해도 되지만
      `/etc/systemd/system/ntpd.service.d`에 `override.conf`를
      만들어서 바꿀 설정만 기록해도 좋다.
    * `systemctl edit ntpd` override하련다.
      * `etc/systemd/system/ntpd.service.d` 디렉토리 생성
      * 편집기 런칭
      * 편집기 종료하면 언급한 디렉토리에 `override.conf` 저장
    * `systemctl edit ntpd --full` 바로 편집 하련다.
  * `systemd-analyze` 부팅시간정보 알려줘
  * `systemd-analyze blame` 각 서비스별 초기화 시간 알려줘
  * `systemd-analyze plot > a.html` html export
  * `systemd-analyze critical-chain` 시간을 많이 소요하는 서비스들을 트리형태로 보여줘
  * `systemd-analyze critical-chain firewalld.service`
  * `systemctl --failed` 실패한 녀석들 보여줘
  * `systemctl rescue` `systemctl isolate runlevel3.target` `systemctl isolate grphical.target`
    * 타겟은 동시에 서비스들을 시작하는 걸 허용하기 위한 그룹 메커니즘이다. 
  * `systemctl set-default grpahical.target` 타겟 바꾸기
  * `systemctl get-default`

## Automation

* `expect`
  * interactive job 을 프로그래밍할 수 있다.
  * [expect 를 이용한 자동화 프로그래밍](https://www.joinc.co.kr/w/Site/JPerl/expect)
  * [Hak5 - Automate Everything, Using Expect, Hak5 1023.1](https://www.youtube.com/watch?v=dlwqyMW5H5I)

## oneline commands

* 파일의 내용을 정렬하고 집합연산해 보자.

```bash
cat a.txt b.txt | sort | uniq > c.txt   # c is a union b
cat a.txt b.txt | sort | uniq -d > c.txt   # c is a intersect b
cat a.txt b.txt | sort | uniq -u > c.txt   # c is set difference a - b
```

* 파일을 읽어서 특정한 열을 더해보자.

```bash
awk '{ x += $3 } END { print x }' a.txt
```

* 모든 디렉토리의 파일들을 재귀적으로 크기와 날짜를 출력해보자. `ls -lR` 과 같지만 출력형식이 더욱 간단하다. 

```bash
find . -type f -ls
```

* `acc_id` 에 대해 얼마나 많은 요청이 있었는지 알아보자.

```bash
cat access.log | egrep -o 'acct_id=[0-9]+' | cut -d= -f2 | sort | uniq -c | sort -rn
```

* 파일의 변경을 모니터링 해보자.

```bash
# 디렉토리의 변경을 모니터링
watch -d -n 2 'ls -rtlh | tail'
# 네트워크 변경을 모니터링
watch -d -n 2 ifconfig
```

* 마크다운을 파싱하고 임의의 것을 출력한다.

```bash
function taocl() {
  curl -s https://raw.githubusercontent.com/jlevy/the-art-of-command-line/master/README.md |
    pandoc -f markdown -t html |
    xmlstarlet fo --html --dropdtd |
    xmlstarlet sel -t -v "(html/body/ul/li[count(p)>0])[$RANDOM mod last()+1]" |
    xmlstarlet unesc | fmt -80
}
```

# Security

## root 소유의 setuid, setgid 파일 검색 후 퍼미션 조정하기

owner 가 root 인 파일들을 생각해보자. setuid 가 설정되어 있으면 실행 되었을 때 EUID 가 root 로 변경된다. 매우 위험하다. 그러한 파일들을 찾아서 위험해 보인다면 권한을 변경해준다.

```bash
find / -user root -perm 4000 -print
```

# System Monitoring

## swapin, swapout

process 의 virtual memory 는 page(4KB) 단위로 잘게 나뉘 어져
있다. page 들은 물리 메모리에 옮겨질때 꼭 연속적이지 않아도
된다. pagetable 에 의해 logical address 가 physical address 로 잘
변환되기 때문이다. [참고](https://www.slideshare.net/sunnykwak90/ss-43933481)

물리 메모리에 적재된 프로세스의 메모리 공간 전체를 디스크의 스왑
영역에 일시적으로 내려 놓는 것을 swapping 이라고 한다. 덜 중요한
프로세스는 더 중요한 프로세스를 위해 물리메모리에서 swapspace 로
swapping 되야 한다. 디스크에서 물리메모리로 프로세스를 옮기는 작업을
swap-in 이라고 한다. 물리 메모리에서 디스크로 프로세스를 옮기는 작업을
swap-out 이라고 한다.

swap-in, swap-ou 의 횟수가 많다면 물리 메모리가 부족하다는 의미이다.
