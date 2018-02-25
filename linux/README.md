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
  - [메뉴얼](#%EB%A9%94%EB%89%B4%EC%96%BC)
  - [자주 사용](#%EC%9E%90%EC%A3%BC-%EC%82%AC%EC%9A%A9)
  - [유저 관리](#%EC%9C%A0%EC%A0%80-%EA%B4%80%EB%A6%AC)
  - [파일 권한](#%ED%8C%8C%EC%9D%BC-%EA%B6%8C%ED%95%9C)
  - [시스템 모니터링](#%EC%8B%9C%EC%8A%A4%ED%85%9C-%EB%AA%A8%EB%8B%88%ED%84%B0%EB%A7%81)
  - [로그](#%EB%A1%9C%EA%B7%B8)
  - [텍스트](#%ED%85%8D%EC%8A%A4%ED%8A%B8)
  - [디버깅](#%EB%94%94%EB%B2%84%EA%B9%85)
  - [압축](#%EC%95%95%EC%B6%95)
  - [에디터](#%EC%97%90%EB%94%94%ED%84%B0)
  - [데몬 관리](#%EB%8D%B0%EB%AA%AC-%EA%B4%80%EB%A6%AC)
- [Security](#security)
  - [root 소유의 setuid, setgid파일 검색 후 퍼미션 조정하기](#root-%EC%86%8C%EC%9C%A0%EC%9D%98-setuid-setgid%ED%8C%8C%EC%9D%BC-%EA%B2%80%EC%83%89-%ED%9B%84-%ED%8D%BC%EB%AF%B8%EC%85%98-%EC%A1%B0%EC%A0%95%ED%95%98%EA%B8%B0)
- [System Monitoring](#system-monitoring)
  - [swapin, swapout](#swapin-swapout)


-----

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
  * 네트워크 상태좀 알려다오
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
  * socket statistics. netstat과 옵션의 의미가 유사하다.
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
* `vmstat`
  * virtual memory 통계 보여조
  * `vmstat 1`
  * `vmstat -s` 부트이후 통계
  * `vmstat -S` 스와핑 통계
  * `vmstat -i` 장치당 인터럽트

| 범주   | 필드 이름 | 설명                                                                                              |
|--------|-----------|---------------------------------------------------------------------------------------------------|
| procs  | r         | 전달 대기열의 커널 스레드 수                                                                      |
|        | b         | 리소스를 대기 중인 차단된 커널 스레드 수                                                          |
|        | w         | 처리 중인 리소스 완료를 대기 중인 스왑 아웃된 LWP 수                                              |
| memory |           | 실제 메모리 및 가상 메모리의 사용에 대해 보고합니다.                                              |
|        | swap      | 사용 가능한 스왑 공간                                                                             |
|        | free      | 해제 목록의 크기                                                                                  |
| page   |           | 페이지 폴트 및 페이지 작업을 초 단위로 보고합니다.                                                |
|        | re        | 재확보된 페이지                                                                                   |
|        | mf        | 작은 결함 및 큰 결함                                                                              |
|        | pi        | 페이징인(킬로바이트)                                                                              |
|        | po        | 페이징아웃(킬로바이트)                                                                            |
|        | fr        | 해제됨(킬로바이트)                                                                                |
|        | de        | 최근 스왑 인된 프로세스에서 필요한 예상 메모리                                                    |
|        | sr        | 현재 사용되고 있지 않은 page 데몬으로 스캔된 페이지. sr이 0이 아니면 page 데몬이 실행된 것입니다. |
| disk   |           | 최대 4개 디스크의 데이터에 대한 초당 디스크 작업 수를 보고합니다.                                 |
| faults |           | 초당 트랩/인터럽트 비율을 보고합니다.                                                             |
|        | in        | 초당 인터럽트                                                                                     |
|        | sy        | 초당 시스템 호출                                                                                  |
|        | cs        | CPU 컨텍스트 전환 비율                                                                            |
| cpu    |           | CPU 시간 사용을 보고합니다.                                                                       |
|        | us        | 사용자 시간                                                                                       |
|        | sy        | 시스템 시간                                                                                       |
|        | id        | 유휴 시간                                                                                         |

* `free`
  * physical memory와 swap memory의 상태를 알려다오
  * `free -b` `free -k` `free -m` `free -g`
  * `free -t` total 추가해조
  * `free -o` buffer adjusted line은 빼고 보여다오
  * `free -s 5` 5초마다 갱신해서 보여줘
  * `free -l` low, high size 통계보여줘
  * [free example](http://www.golinuxhub.com/2015/02/what-is-bufferscache-in-free-command-in.html)
    * physical memory = 2048
      * 2048 = 1869 + 152 (부팅때문에 커널에 의해 예약된 공간)
    * free가 471이지만 buffer/cache adjust line을 통해 895까지 확장 될 수 있음을 알 수 있다.

```
# free -m
             total       used       free     shared    buffers     cached
Mem:          1869       1398        471          0         24        400
-/+ buffers/cache:        974        895
Swap:         3999          0       3999
```

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
  * `sudo vi /etc/default/sysstat` `ENABLED="true"` 매 10분마다 데이터 수집을 위한 sadc활성화
    * `sudo vi /etc/cron.d/sysstat` `* * * * * root command -v debian-sa1 > /dev/null && debian-sa1 1 1` 매분마다 해볼까
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

| class        | 이름  | 설명                                                                                                     |
|--------------|-------|:--------------------------------------------------------------------------------------------------------:|
| load average |       | 1분, 5분, 15분동안 run queue에 저장된 job의 평균개수이다. 1이면 여유 5이면 버겁고 10이상이면 과부하이다. |
| cpu          | us    | user 용 processor 활용 비율                                                                              |
|              | sy    | system 용 processor 활용 비율                                                                            |
|              | id    | idle 용 processor 활용 비율                                                                              |
| PhysysMem    | wired | non-paged pool???                                                                                        |

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
  * `date --date="12/2/2014"`
  * `date --date="next mon"`
  * `date --date=@5` UTC 이후로 5초 지났다.
  * `date --date='3 seconds ago'`
  * `date +%<format-option>`
* `time`
  * command 실행하고 소요시간 출력해다오
  * `time a.out`
* `du`
  * `du -h /home/iamslash`
  * `du -sh /home/iamslash` 요약해서 알려줘
  * `du -ah /home/iamslash` 모든 파일과 디렉토리 알려줘
  * `du -hk /home/iamslash`
  * `du -hm /home/iamslash`
  * `du -hc /home/iamslash` 마지막에 total보여줘
  * `du -ah --exclude="*.txt /home/iamslash"`
  * `du -ah --time /home/iamslash`
* `df`
  * `df -h` free space를 읽기쉬운 형식으로 보여줘
  * `df -ah` 모든 파일 시스템에 대한 free space를 읽기쉬운 형식으로 알려다오
  * `df -ih` i-node정보좀 털어봐라
  * `df -Th` 파일 시스템 타입좀
  * `df -th ext3` ext3포함해서 알려줘
  * `df -xh ext3` ext3빼고 알려줘
* `uname`
  * 운영체제이름 알려다오
  * `uname -a`
* `hostname`
  * 호스트이름 알려다오
* `last`
  * 마지막에 로그인했던 user, tty, host알려다오
* `uptime`
  * 시스템이 실행된지 얼마나 지났니?
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
  * `grep -i "the" a.txt` case insensitive
  * `grep "lines.*empty" a.txt` regular expression
  * `grep -i "is" a.txt`
  * `grep -iw "is" a.txt` 검색의 대상은 단어이다.
  * `grep -A 3 -i "example" a.txt` 검색된 줄 이후 3줄 더 보여줘라
  * `grep -B 2 "single WORD" a.txt` 검색된 줄 이전 2줄 더 보여줘라
  * `grep -C 2 "example" a.txt` 검색된 줄 이전 이후 2줄 더 보여줘라
  * `export GREP_OPTIONS='--color=auto' GREP_COLOR='100;8'` `grep this a.txt`
  * `grep -r "this" *`
  * `grep -v "go" a.txt` 검색 대상이 안된 줄을 보여줘라
  * `grep -c "go" a.txt` 검색된 수를 보여다오
  * `grep -l this a.*` 검색된 파일들을 보여다오
  * `grep -o "is.*line" a.txt` 검색된 문자열들만 보여다오
  * `grep -o -b "3" a.txt` 검색된 위치를 보여다오
  * `grep -n "go" a.txt` 검색된 줄번호도 보여다오
* `xargs`
  * 구분자 `\n`을 이요해서 argument list를 구성하고 command 한개에 argument 하나씩 대응해서 실행하자
  * xargs 가 실행할 command가 없다면 `/bin/echo`를 사용한다.
  * `echo a b c d e f | xargs`
  * `echo a b c d e f | xargs -n 3` argument는 3개씩 한조가 되라
  * `echo a b c d e f | xargs -p -n 3` prompt 등장
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
  * `tail -f /var/log/messages`
  * `tail -f /tmp/a.log --pid=2575` 2575프로세스 죽을때까지
  * `tail -f /tmp/debug.log --retry`
* `head`
  * `head -n 5 a.txt`
  * `head -4 a.txt`
  * `head -n -5 a.txt`
  * `ls | head`
* `awk`
  * [awk](../awk/README.md)
* `uniq`
  * `uniq a.txt`
  * `uniq -c a.txt`
  * `uniq -d a.txt` 중복된 녀석들은 제거해라.
  * `uniq -D a.txt` 중복된 녀석들은 제거하지마라
  * `uniq -u a.txt` 중복되지 않은 행들은?
  * `uniq -c -w 8 a.txt` 처음 8바이트만 중복되지 않은 녀석들을 보여다오
  * `uniq -D -w 8 a.txt` 처음 8바이트만 중복된 녀석들을 보여다오
  * `uniq -D -s 2 a.txt` 2바이트는 건너띄고 중복된 녀석들을 보여다오
  * `uniq -D -f 2 a.txt` 2필드는 건너띄고 중복된 녀석들을 보여다오
* `sort`
  * `sort a.txt > b.txt`
  * `sort -r a.txt > b.txt`
  * `sort -nk2 a.txt` 2열을 기준으로 정렬해라.
  * `sort -k9 a.txt`
  * `ls -l /home/$USER | sort -nk5`
  * `sort -u a.txt` 정렬하고 중복된 거 지워라
  * `sort a.txt b.txt`
  * `sort -u a.txt b.txt`
  * `ls -l /home/$USER | sort -t "," -nk2,5 -k9` 숫자인 2, 5열, 문자인 9열을 기준으로 정렬해라.x
* `wc`
  * `wc a.txt` 줄개수, 단어수, 바이트 표시해조
  * `wc -l a.txt` 줄개수 보여줘
  * `wc -w a.txt` 단어수 보여줘
  * `wc -c a.txt` `wc -m a.txt` 바이트수 보여도
  * `wc -L a.txt` 가장 길이가 긴 행의 문자개수는?

## 텍스트

* `paste`
  * 파일의 특정 행들을 머지해줘
  * `paste a.txt`
  * `paste -s a.txt` 모든 행을 join해도
  * `paste -d, -s a.txt` comma를 구분자로 모든 행을 join해도
  * `paste - - < a.txt` 2열로 merge해라.
  * `paste -d':' - - < a.txt` 구분자는 `:`로 하고 2열로 merge해라.
  * `paste -d ':,' - - - < a.txt` 구분자는 `:` `,` 로 하고 3열로 merge해라.
  * `paste a.txt b.txt` 파일 두개 2열로 merge하라.
  * `paste -d, a.txt b.txt` 구분자는 `,` 으로 파일 두개 2열로 merge해라.
  * `cat b.txt | paste -d, a.txt -`
  * `cat a.txt | paste -d, - b.txt`
  * `cat a.txt b.txt | paste -d, - -`
  * `paste -d'\n' a.txt b.txt`
* [sed](../sed/)
* `tr`
  * 문자열 번역해줘
  * `echo HELLO | tr [A-Z] [a-z]`
  * `tr '{}' '()' < a.txt > b.txt`
  * `echo "the geek stuff" | tr -d 't'` t문자 지워줘
  * `echo "my username is 432234 | tr -d [:digit:]` 숫자 지워줘
  * `echo "my username is 432234 | tr -cd [:digit:]` 숫자 빼고 지워줘
  * `tr -cd [:print:] < a.txt` non printable 문자들을 지워다오
  * `tr -s '\n' ' ' < a.txt` 모든 행을 join하자.
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

## 데몬 관리

* cron
  * 반복 예약 작업 등록
  * `crontab -l` 등록된 목록을 보여줘
  * `crontab -l -u iamslash` iamslash USER의 목록 보여줘
  * `crontab -e` 크론탭 수정해볼까
    * `* * * * * /tmp/a.sh` 매 1분 마다 실행해 
    * `15,45 * * * * /tmp/a.sh` 매시 15, 45분 마다 실행해 
    * `*/10 * * * * /tmp/a.sh` 10분 마다 실행해 
    * `0 2 * * * /tmp/a.sh` 매일 02:00에 마다 실행해 
    * `30 */6 * * * /tmp/a.sh` 매 6시간 마다(00:30, 06:30, 12:30, 18:30) 실행해 
    * `30 1-23/6 * * * /tmp/a.sh` 1tlqnxj 매 6시간 마다(01:30, 07:30, 13:30, 19:30) 실행해 
    * `0 8 * * 1-5 /tmp/a.sh` 평일(월-금) 08:00 
    * `0 8 * * 0,6 /tmp/a.sh` 주말(일,토) 08:00
  * `crontab -r` 모두 삭제

* systemd
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

# Security

## root 소유의 setuid, setgid파일 검색 후 퍼미션 조정하기

setuid가 설정되어 있으면 실행 되었을 때 EUID가 root로 변경된다. 불필요한
파일을 찾아서 퍼미션을 변경하자.

```bash
find / -user root -perm 4000 -print
```
# System Monitoring

## swapin, swapout

process의 virtual memory는 page(4KB)단위로 잘게 나뉘 어져
있다. page들은 물리 메모리에 옮겨질때 꼭 연속적이지 않아도
된다. pagetable에 의해 logical address가 physical address로 잘
변환되기 때문이다. [참고](https://www.slideshare.net/sunnykwak90/ss-43933481)

물리 메모리에 적재된 프로세스의 메모리 공간 전체를 디스크의 스왑
영역에 일시적으로 내려 놓는 것을 swapping이라고 한다. 덜 중요한
프로세스는 더 중요한 프로세스를 위해 물리메모리에서 swapspace로
swapping되야 한다. 디스크에서 물리메모리로 프로세스를 옮기는 작업을
swap-in이라고 한다. 물리 메모리에서 디스크로 프로세스를 옮기는 작업을
swap-out이라고 한다.

swap-in(page-in), swap-out(page-out)이 많다면 물리 메모리가 모자르다는
의미이다.
