- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Linux Application Source Code](#linux-application-source-code)
- [Permissions](#permissions)
  - [Mode](#mode)
    - [Setuid](#setuid)
    - [Setgid](#setgid)
    - [Sticky Bit](#sticky-bit)
- [Special Directories](#special-directories)
- [Special Files](#special-files)
- [Speicial FileSystem](#speicial-filesystem)
- [Pipe](#pipe)
- [Package Managers](#package-managers)
  - [apt-get](#apt-get)
  - [apt](#apt)
  - [dpkg](#dpkg)
  - [brew](#brew)
  - [yum](#yum)
- [Commands](#commands)
  - [Manual](#manual)
  - [Basic](#basic)
  - [Job Management](#job-management)
  - [Process Management](#process-management)
  - [User Management](#user-management)
  - [File Permissions](#file-permissions)
  - [System Monitoring](#system-monitoring)
  - [Logs](#logs)
  - [Text](#text)
  - [Debugging](#debugging)
  - [Compressions](#compressions)
  - [Editors](#editors)
  - [Files](#files)
  - [Daemon Management](#daemon-management)
  - [Disk](#disk)
  - [Network](#network)
  - [Automation](#automation)
  - [Oneline Commands](#oneline-commands)
  - [Tips](#tips)
- [Exit Codes](#exit-codes)
- [Performance Monitoring](#performance-monitoring)
- [Security](#security)
  - [root 소유의 setuid, setgid 파일 검색 후 퍼미션 조정하기](#root-소유의-setuid-setgid-파일-검색-후-퍼미션-조정하기)
- [System Monitoring](#system-monitoring-1)
  - [Load Average](#load-average)
  - [Swapin, Swapout](#swapin-swapout)
  - [Memory](#memory)
- [Network Kernel Parameters](#network-kernel-parameters)
- [File Kernel Parameters](#file-kernel-parameters)
- [Cgroup](#cgroup)
- [Slab](#slab)

-------------------------------------------------------------------------------

# Abstract

linux 를 정리한다. [bash](/bash/), [awk](/awk/), [sed](/sed/) 도 중요하다.

systemd 가 설치된 [ubuntu docker
image](https://hub.docker.com/r/jrei/systemd-ubuntu) 를 이용하여 실습하자.

```bash
$ docker run -d --name systemd-ubuntu --privileged jrei/systemd-ubuntu
$ docker exec -it systemd-ubuntu bash
```

[Linux Command
MindMap](https://xmind.app/m/WwtB/?utm_source=substack&utm_medium=email) 는
Linux Command 대부분을 MindMap 으로 표현했다. 

# References

*  [Linux Command MindMap](https://xmind.app/m/WwtB/?utm_source=substack&utm_medium=email)
* [DevOps와 SE를 위한 리눅스 커널 이야기](https://aidanbae.github.io/article/book/linux/)
  * DevOps 를 위한 linux kernel 지식

# Materials

- [리눅스 모의고사](https://tech.osci.kr/%EC%8B%A4%EC%A0%84-linux-%EB%AA%A8%EC%9D%98%EA%B3%A0%EC%82%AC/)
* [Site Reliability Engineer (SRE) Interview Preparation Guide](https://github.com/mxssl/sre-interview-prep-guide)
* [리눅스 엔지니어 기술 면접 질문지](https://docs.google.com/document/u/0/d/1WE1V4uczxavqLY-nyr3qNqCxqzoOf8Vg6Z-Lf0c3DwU/mobilebasic)
  * [리눅스 엔지니어 기술 면접 질문지 | github](https://github.com/pjhwa/linux-engineers/wiki)
* [리눅스 서버를 다루는 기술](https://thebook.io/006718/)
  * 최고의 ubuntu 입문서
* [The Art of Command Line | github](https://github.com/jlevy/the-art-of-command-line/blob/master/README-ko.md)
  * 커맨드 라인들 소개
* [Most Important Penetration Testing commands Cheat Sheet for Linux Machine](https://techincidents.com/important-penetration-testing-cheat-sheet/)
  * 유용한 시스템 침입 테스트 커맨드들
* [command line reference](https://ss64.com/)
  * bash, macOS, cmd, powershell 등등의 command line reference

# Linux Application Source Code

* [coreutils | github](https://github.com/coreutils/coreutils)
  * ls, mkdir, nohup, pwd, rm, sleep, tail, tee, wc, whoami, cat, cp, cut, df,
    du, echo, head, copy, chown, basename, etc...
* [procps | gitlab](https://gitlab.com/procps-ng/procps)
  * procps is a set of command line and full-screen utilities that provide
    information out of the pseudo-filesystem most commonly located at /proc.
  * free, kill, pgrep, pkill, pmap, ps, pwdx, skill, slabtop, snice, sysctl,
    tload, top, uptime, vmstat, w, watch
* [bash | savannah](http://git.savannah.gnu.org/cgit/bash.git/)
* [psmisc | gitlab](https://gitlab.com/psmisc/psmisc)
  * fuser - Identifies processes using files or sockets
  * killall - kills processes by name, e.g. killall -HUP named
  * prtstat - prints statistics of a process
  * pslog - prints log path(s) of a process
  * pstree - shows the currently running processes as a tree
  * peekfd - shows the data travelling over a file descriptor

# Permissions

## Mode

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

mode 는 8 진법으로 표기했을때 왼쪽 부터 **특수권한**, **유저권한**, **그룹권한**, **기타권한**과 같이 구성된다. 각 권한 별로 3비트가 할당된다. **특수권한**의 3 비트는 왼쪽부터 **setuid**, **setgid**, **stckybit** 을 의미하고 **유저권한**, **그룹권한**, **기타권한**의 3비트는
왼쪽부터 **읽기**, **쓰기**, **실행** 권한을 의미한다.

특수권한을 확인 하는 요령은 다음과 같다. `ls -l` 했을때 setuid 가 on 되어 있으면 유저권한의 3 비트중 가장 오른쪽 비트가 `s` 혹은 `S` 로 보여진다. setgid 가 on 되어 있으면 그룹권한의 3 비트중 가장 오른쪽 비트가 `s` 혹은 `S` 로 보여진다. stickybit 가 on 되어 있으면 기타권한의 3 비트중 가장 오른쪽 비트가 `t` 혹은 `T` 로 보여진다. 표시되는 권한의 실행권한이 없다면 소문자로 보여지고 실행권한이 있다면 대문자로 보여진다.

### Setuid

`setuid` 가 설정된 파일을 실행할때 다음과 같은 현상이 발생한다. 실행을 위해 태어난 프로세스의 `EUID`(유효 사용자 아이디)가 `RUID`(실행 사용자 아이디)에서 파일의 소유자 아이디로 변경된다. 실행순간만 권한을 빌려온다고 이해하자.

### Setgid

`setgid` 가 설정된 파일을 실행할때 다음과 같은 현상이 발생한다. 실행을 위해 태어난 프로세스의 `EGID`(유효 그룹 아이디)가 `RGID`(실행 그룹
아이디)에서 파일의 소유 그룹 아이디로 변경된다. 실행순간만 권한을 빌려온다고 이해하자.

### Sticky Bit

linux 는 파일의 sticky bit 를 무시한다. 디렉토리에 sticky bit 가 설정되어 있다면 누구나 해당 디렉토리에서 파일을 생성할 수 있지만 삭제는 디렉토리 소유자, 파일 소유자, 슈퍼 유저만 할 수 있다. 그래서 sticky bit 를 공유모드라고 한다.

# Special Directories

| DIRECTORY | DESCRIPTION |
| - | - |
| `/`         | / also know as “slash” or the root. |
| `/bin`        | Common programs, shared by the system, the system administrator and the users. |
| `/boot`       | Boot files, boot loader (grub), kernels, vmlinuz |
| `/dev`        | Contains references to system devices, files with special properties. |
| `/etc`        | Important system config files. |
| `/home`       | Home directories for system users. |
| `/lib`        | Library files, includes files for all kinds of programs needed by the system and the users. |
| `/lost+found` | Files that were saved during failures are here. |
| `/mnt`        | Standard mount point for external file systems. |
| `/media`      | Mount point for external file systems (on some distros). |
| `/net`        | Standard mount point for entire remote file systems ? nfs. |
| `/opt`        | Typically contains extra and third party software. |
| `/proc`       | A virtual file system containing information about system resources. |
| `/root`       | root users home dir. |
| `/sbin`       | Programs for use by the system and the system administrator. |
| `/tmp`        | Temporary space for use by the system, cleaned upon reboot. |
| `/usr`        | Programs, libraries, documentation etc. for all user-related programs. |
| `/var`        | Storage for all variable files and temporary files created by users, such as log files, mail queue, print spooler. Web servers, Databases etc. |

# Special Files

| DIRECTORY                   | DESCRIPTION                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| `/etc/passwd`                 | Contains local Linux users.                                         |
| `/etc/shadow`                 | Contains local account password hashes.                             |
| `/etc/group`                  | Contains local account groups.                                      |
| `/etc/init.d/`                | Contains service init script ? worth a look to see whats installed. |
| `/etc/hostname`               | System hostname.                                                    |
| `/etc/network/interfaces`     | Network interfaces.                                                 |
| `/etc/resolv.conf`            | System DNS servers.                                                 |
| `/etc/profile`                | System environment variables.                                       |
| `~/.ssh/`                     | SSH keys.                                                           |
| `~/.bash_history`             | Users bash history log.                                             |
| `/var/log/`                   | Linux system log files are typically stored here.                   |
| `/var/adm/ `                  | UNIX system log files are typically stored here.                    |
| `/var/log/apache2/access.log` | Apache access log file typical path.                                |
| `/var/log/httpd/access.log`   | Apache access log file typical path.                                |
| `/etc/fstab`                  | File system mounts.                                                 |

# Speicial FileSystem

* [What is “udev” and “tmpfs”](https://askubuntu.com/questions/1150434/what-is-udev-and-tmpfs)
* [Specfs, Devfs, Tmpfs, and Others](https://www.linux.org/threads/specfs-devfs-tmpfs-and-others.9382/)

----

* `df`

```bash
$ df -hT
Filesystem      Size  Used Avail Use% Mounted on
overlay          59G  7.3G   49G  14% /
tmpfs            64M     0   64M   0% /dev
tmpfs          1000M     0 1000M   0% /sys/fs/cgroup
shm              64M     0   64M   0% /dev/shm
/dev/sda1        59G  7.3G   49G  14% /etc/hosts
tmpfs          1000M     0 1000M   0% /proc/acpi
tmpfs          1000M     0 1000M   0% /sys/firmware
```

| file system | DESCRIPTION                                                                                                     |
| ----------- | -- |
| `tmpfs`     | Virtual filesystem located in RAM |
| `udev`      | Virtual filesystem related to the devices files, aka the interface between actual physical device and the user. |
| `/dev/sda1` | Device file |
| `/dev/sdb`  | Device file |

# Pipe

Bash의 파이프(pipe)는 한 명령의 출력을 다른 명령의 입력으로 전달하는 방법입니다. 이를 통해 여러 명령어를 결합하여 강력한 기능을 수행하고, 프로세스 간 데이터의 흐름을 연결할 수 있습니다. 파이프는 '|' 기호를 사용하여 표시됩니다.

예시:

두 명령어를 결합하기:

```bash
$ ls | sort
```

이 명령은 'ls' 명령의 출력을 'sort' 명령의 입력으로 전달합니다. 따라서 현재 디렉토리의 파일 목록이 사전식으로 정렬되어 출력됩니다.

여러 명령어를 파이프를 사용하여 결합하기:

```bash
$ cat some_file.txt | grep "search_term" | wc -l
```

이 명령은 'cat some_file.txt' 명령의 출력을 'grep "search_term"' 명령의 입력으롷 전달합니다. 'grep' 명령은 주어진 검색어에 해당하는 줄들만 출력합니다. 그리고 그 출력은 'wc -l' 명령의 입력으로 전달되어, 결과적으로 검색어에 해당하는 줄의 개수가 출력됩니다.

파이프를 사용하면 명령어의 결과를 쉽게 체이닝(chain)하여 프로세스 간의 데이터 통신을 구현할 수 있으며, 작업을 간소화할 수 있습니다.

# Package Managers

## apt-get

* [apt-get(8) - Linux man page](https://linux.die.net/man/8/apt-get)
* `/etc/apt/sources.list` 에 기재된 repo 이 packages 를 설치한다.
  * `main` : Free SW Ubuntu officially supports
  * `universe` : Free SW Ubuntu doesn't support
  * `restricted` : Non-free SW Ubuntu officially supports 
  * `multiverse` : Non-free SW Ubuntu doesn't support

-----

```bash
# Update the list of packages from /etc/apt/sources.list
$ apt-get update
# Upgrade every packages
$ apt-get upgrade
$ apt-get install curl
$ apt-get --reinstall install curl
$ apt-get remove curl
# remove curl package including config files
$ apt-get --purge remove curl
# Remove unused pakcages
$ apt-get autoremove
# Remove downloaded files
$ apt-get clean/autoclean
# Download curl pakcage source
$ apt-get source curl

# Search curl pakcage
$ apt-cache search curl
# Show curl package informations
$ apt-cache show curl
# Show installed package and the location
$ apt-cache policy curl
# Show dependencies of the package
$ apt-cache depends curl
# Show reversed dependencies of the package
$ apt-cache rdepends curl
```

* [How to find out which versions of a package can I install on APT](https://superuser.com/questions/393681/how-to-find-out-which-versions-of-a-package-can-i-install-on-apt)

```bash
$ apt-cache madison vim
```

## apt

* [apt 와 apt-get 의 차이](https://jhhwang4195.tistory.com/69)

-----

`apt-get` 이 먼저 만들어지고 `apt-get, apt-cache` 를 하나의 command 로 처리하기
위해 탄생했음.

```bash
$ apt update/upgrade
$ apt install
$ apt remote
$ apt autoremove
$ apt autoclean
$ apt clean
# Show installed package
$ apt show (=apt-cache show)
# Search package in repo
$ apt search
# Show installed package and the location
$ apt policy
```

## dpkg

* [apt 와 dpkg 의 차이점](https://sung-studynote.tistory.com/78)
* `dpkg` 는 package 의 dependency 를 처리하지 않는다. 그러나 `apt-get` 은 pakcage 의 dependency 를 처리하여 관련된 package 를 모두 설치하고 환경변수 또한 설정한다. 또한 `apt-get` 은 내부적으로 `dpkg` 를 사용한다. 

-----

```bash
# Install package
$ dpkg -i a.deb
# Uninstall pakcage
$ dpkg -r <pakcage-name>
# Uninstall pakcage including env variables
$ dpkg -P <pakcage-name>
# Show installed pakcage information
$ dpkg -l <pakcage-name>
# Show file names of installed pakcage
$ dpkg -L <pakcage-name>
```

## brew

```bash
$ brew install curl
```

## yum

```bash
$ yum install curl
```

# Commands

> [The Art of Command Line | github](https://github.com/jlevy/the-art-of-command-line/blob/master/README-ko.md)

Application commands 와 bash builtin commands 등이 있다. 상황별로 유용한 commands 를 정리한다. bash builtin commands 의 경우 `/usr/bin/` 에 application commands 으로 존재한다. 다음은 macOS 에서 `/usr/bin/ulimit` 의 내용이다. 단지 bash builtin 으로 command 와 argument 들을 전달 하고 있다.

```bash
builtin `echo ${0##*/} | tr \[:upper:] \[:lower:]` ${1+"$@"}
```

- `echo ${0##*/}` 는 `${0}` 에서 마지막 `/` 이후 application name 을 제외하고 모두 지운다. [Bash Shell Parameter Expansion](/bash/README.md#shell-parameter-expansion) 참고.
- `${1+"$@"}` 는 1-th arg 를 포함한 모든 args 를 의미한다.

## Manual

* `man`
  * 메뉴얼 좀 보여줘봐
  * `man ls`
  * `man man`
* `apropos`
  * 잘은 모르겠고 이런거 비슷한 거 찾아줘봐라
  * `man -k` 와 같다.
  * `apropos brew`
* `info`
  * 메뉴얼 좀 보여줘봐. 단축키는 emacs 와 비슷한데?
  * `info ls`

## Basic

* `history`
  * 최근에 사용한 command line 보여다오.
  * `history` 
  * `!ssh` 
  * `!!` 
  * `!14`
* `ls`
  * 디렉토리들과 파일들을 보여다오.
  * `ls -al`
* `cd`
  * 작업디렉토리를 바꿔다오.
  * `cd /usr/bin`
  * `cd ~`
* `pwd`
  * 현재 작업디렉토리는 무엇이냐.
* `pushd, popd`
  * 디렉토리를 스택에 넣고 빼자.
    ```bash
    $ pushd /usr/bin
    /usr/bin 
    $ pushd /home
    /home /usr/bin
    $ popd
    /usr/bin
    $ pwd
    /usr/bin
    ```
* `ln`
  * 심볼릭 링크 만들어 다오.
  * `ln -s /Users/iamslash/demo /Users/iamslash/mylink`
* `cp`
  * 복사해 다오.
  * `cp -r a b`
* `mv`
  * 파일을 옮기거나 파일 이름 바꿔다오.
  * `mv a b`
* `rm`
  * 파일 혹은 디렉토리를 지워다오.
  * `rm -rf *`
* `cat`
  * 파일을 이어 붙이거나 출력해 다오.
  * `cat a.txt`
  * `cat a.txt b.txt > c. txt`
* `more`
  * 한번에 한화면씩 출력해 다오.
  * `man ls | more`
* `less`
  * `more` 보다 기능이 확장 된 것.
  * `man ls | less`
* `echo`
  * 화면에 한줄 출력해 다오.
  * `echo hello` `echo $A`
* `touch`
  * 파일의 최종 수정 날짜를 바꾸자.
  * `touch a.txt`
* [diff](/diff/README.md)
  * `diff a.txt b.txt` 두개의 파일 비교
  * `diff -r /tmp/foo /tmp/bar` 두개의 디렉토리를 비교
* `which`
  * command 위치는 어디있어?
  * `which ls`
  * `command -v ls`
* `file`
  * 이 파일의 종류는 무엇?
  * `file a.out`
* `ps`
  * 현재 실행되고 있는 프로세스들의 스냅샷을 보여다오
  * option 의 종류는 UNIX, BSD, GNU 와 같이 3 가지이다. UNIX 는 `-` 를 사용한다.
    GNU 는 `--` 를 사용한다.
  * `ps a` Lift the BSD-style "only yourself" restriction. "only yourself" 조건을 제거하라.
  * `ps u` Display user-oriented format.
  * `ps x` Lift the BSD-style "must have a tty" restriction. "must have a tty" 조건을 제거하라.
    * [tty](https://www.howtoforge.com/linux-tty-command/)s stands for teletype or terminal.
  * `ps j` BSD job control format.
  * `ps f` ASCII art process hierarchy (forest).
  * `ps m` Show threads after processes.
  * `ps aux` lift "only yourself", display user-oriented format, lift "must have a tty".
  * `ps axjf` show processes with the tree.
  * `ps axms` show processes with threads.
* `kill`
  * [How to kill Processes in Linux using kill, killall and pkill](https://www.thegeekdiary.com/how-to-kill-processes-in-linux-using-kill-killall-and-pkill/)
  * The kill command sends signal 15, the terminate signal, by default.
  * `kill -l`
    * 가능한 시글널 목록을 보여다오
  * `kill -9 123`
    * 123 프로세스에게 SIGKILL 보내다오
* `killall`
  * 이름을 이용하여 특정 프로세스에게 SIGTERM 을 보내자.
  * `killall a.sh`
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
  * `pstree -s -p 19177` 19177 PID 의 부모를 보여다오
* `telnet`
  * TELNET protocol client
* `nc` netcat
  * [넷캣(Netcat) 간단한 사용방법](https://devanix.tistory.com/307)
  * `nc -l -p 1234`
    * simple tcp listen server
  * `nc 127.0.0.1 1234`
    * connect tcp server and send a data
  * `nc -l -u 1234`
    * simple udp listen server
  * `nc -u 127.0.0.1 1234`
    * send a udp data
  * `ps auxf | nc -w3 10.0.2.15 1234`
    * `ps auxf` 결과를 전송하자.
    * `-w3` 를 사용하여 3 초 동안 전송해 보자.
  * `nc -n -v -z -w 1 10.0.2.100 1-1023`
    * `-n` : DNS 이름 주소 변환 안한다.
    * `-v` : verbose
    * `-z` : 데이터 전송 안한다.
    * `-w` : 최대 1 초의 연결 시간
    * `1-1023` : TCP 를 사용해서 1-1023 사이의 포트를 스캔한다.
* `ssh-keygen`
  * [ssh-keygen](https://www.ssh.com/ssh/keygen)
  * `ssh-keygen -t rsa -b 2048`
* `ssh`
  * OpenSSH SSH client
  * `ssh iamslash@a.b.com`
  * `ssh iamslash@a.b.com ls /tmp/doc` Just execute command without allocating terminal
  * `ssh -i ~/.ssh/id_rsa.iamslash iamslash@a.b.com` Use specific private key file
  * `ssh -p 122 iamslash@a.b.com` Use specific port
  * `ssh -T iamslash@a.b.com` Just test without allocating terminal
  * `ssh -p 122 -o StrictHostKeyChecking=no iamslash@a.b.com` Without yes/no of checking host
* `ssh-add`
  * [ssh-copy-id](https://www.ssh.com/ssh/copy-id)
  * 인증 에이전트에게 비밀키를 추가해 놓자.
  * `eval $(ssh-agent -s)`
  * `ssh-add ~/.ssh/id_rsa` 암호 입력을 매번 하지 않기 위해 키체인에 등록하자
* `ssh-copy-id`
  * public key 를 remote 에 복사한다. remote `/etc/ssh/sshd_config` should include `PasswordAuthentication yes`.
  * `ssh-copy-id -p 122 -i ~.ssh/id_rsa.pub admin@xxx.xxx.xxx.xxx`
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
  * terminal multiplexer
  * [tmux](/tmux/)
* `nslookup`
  * domain 을 주고 ip 를 얻오오자.
  * `apt-get install dnsutils`
  * [nslookup 명령어 사용법 및 예제 정리](https://www.lesstif.com/pages/viewpage.action?pageId=20775988)
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
  * DNS name server 와 도메인 설정이 완료된 후 DNS 질의 응답이 정상적으로
    이루어지는 지를 확인한다.
  * `apt-get install dnsutils`
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
  * URL 을 활용하여 data 전송하는 program. HTTP, HTTPS, RTMP 등등을 지원한다.
  * [curl 설치 및 사용법 - HTTP GET/POST, REST API 연계등](https://www.lesstif.com/pages/viewpage.action?pageId=14745703)
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
  * `curl -d @a.js -H "Content-Type: application/json" --user-agent "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.14 (KHTML, like Gecko) Chrome/24.0.1292.0 Safari/537.14" http://a.b.com/a`
    * `-d` 를 이용하여 `a.js` 를 읽어서 HTTP POST 데이터로 전송한다.
    * `-H` 를 이용하여 HTTP HEAD 를 설정한다. 여러개의 HEAD 를 전송하고 싶다면 `-H` 를 여러개 사용하라.
    * `--user-agent` 를 이용하여 BROWSER 를 설정한다.
  * [Curl post multiline json data from stdin](https://www.darklaunch.com/curl-post-multiline-json-data-from-stdin.html)
    ```bash
    $ curl https://api.example.com/posts/create --data @- <<EOF
    "title": "Hello world!",
    "content": "This is my first post."
    }
    EOF
    ```
* `wget`
  * web 에서 파일을 다운로드해 다오.
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
  * `EUID`, `EGID` 를 수정하여 SHELL 을 실행하자.
  * `su - root`
* `bc`
  * 계산기 언어
  * `echo "56.8 + 77.7" | bc` 
  * `echo "scale=6; 60/7.02" | bc`
* `reset`
  * 터미널을 초기화 한다. 지저분 해졌을때 사용하면 좋다.
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
* `tcpdump`
  * [Manpage of TCPDUMP](http://www.tcpdump.org/manpages/tcpdump.1.html)
  * `tcpdump -i eth0` show the eth0 status
  * `tcpdump -i eth0 tcp port 80` show the eth0 port 80 packets
    * if you want to see the contens use `nc`
  * `tcpdump -i eth0 src 192.168.0.1` show the eth0 packets have specific src
  * `tcpdump -i eth0 dst 192.168.0.1` show the eth0 packets have specific dst
  * `tcpdump -i eth0 src 192.168.0.1 and tcp port 80` and condition
  * `tcpdump host 192.168.0.1` show packets have specific src, dst
  * `tcpdump net 192.168.0.1/24` can use CIDR
  * `tcpdump tcp` show only tcp
  * `tcpdump udp` show only udp
  * `tcpdump port 3389` show only port 3389
  * `tcpdump -A port 3389` show only port 3389 and print packets in ASCII
  * `tcpdump src port 3389` show only src port 3389
  * `tcpdump dst port 3389` show only dst port 3389
* `ngrep`
  * [man of ngrep](https://linux.die.net/man/8/ngrep)
    * `ngrep [옵션] [매칭할 패턴] [ BPF 스타일 Filter ]`
  * `ngrep -qd eth0 port 80` eth0 인터페이스의 80 port 기록을 조용히 보여줘
  * `ngrep -iqd eth0 port 80` 대소문자 구분없이 보여줘
  * `ngrep -qd eth0 HTTP port 80` eth0 인터페이스의 HTTP 80 port 기록을 조용히 보여줘
  * `ngrep -v -qt host <ip> and not port 80` 검색 조건을 거꾸로 해서 시간과 함께 보여줘
* `su`
  * [리눅스 su, sudo 명령어 사용법 정리 (root 권한 획득 방법)](https://withcoding.com/106)
  * switch user
  * 로그인 하지 않고  다른 계정으로 전환한다.
  * `su` root user 로 전환
  * `su iamslash` iamslash user 로 전환
  * `su - iamslash` SHELL, home directory 도 전환
  * `whoami` 나는 누구인가?
  * `logout` 전환되기 이전 계정으로 돌아온다.
  * `su -c 'apt-get update'` root 권한으로 command line 실행. sudo 와 비슷하다.
* `sudo`
  * [리눅스 su, sudo 명령어 사용법 정리 (root 권한 획득 방법)](https://withcoding.com/106)
  * root 권한을 이용하여 command line 실행
  * `sudo apt-get update`
  * `sudo -s` su 처럼 root user 로 전환한다. home directory 는 유지.
  * `sudo -i` su 처럼 root user 로 전환한다. home directory 도 전환. 
  * `/etc/sudoers` 파일에 지정된 사용자만 sudo 명령을 사용할 수 있다.
    * `sudo visudo` `/etc/sudoers` 를 수정한다.
      ```
      # User privilege specification
      root    ALL=(ALL:ALL) ALL
      iamslash  ALL=(ALL:ALL) ALL
      ```

## Job Management

> [Understanding the job control commands in Linux – bg, fg and CTRL+Z](https://www.thegeekdiary.com/understanding-the-job-control-commands-in-linux-bg-fg-and-ctrlz/)

Linux에서 process 와 job의 차이는 다음과 같습니다

- 프로세스(Process):
  - 자체적인 주소 공간, 메모리, 데이터 및 연산을 가진 운영체제에서 실행되는 독립적인 프로그램 인스턴스입니다.
  - 각 프로세스에는 고유한 프로세스 ID (PID)가 할당됩니다.
  - 프로세서(CPU)에서 실행되며, 다른 프로세스와 통신을 위해 Inter-Process Communication (IPC) 메커니즘을 사용합니다.
  - 부모-자식 구조를 사용하여 생성(parent process)과 종료(child process)가 가능합니다.
- 잡(Job):
  - 실행 중인 셸(Bash and others)의 하위 프로세스입니다.
  - 프로세스가 터미널에서 실행되는 경우 배경(background) 또는 전경(foreground)에서 작업으로 간주할 수 있습니다.
  - Job Control은 사용자가 셸로 돌아와 이전에 수행되었던 일을 다시 시작할 수 있게 해줍니다.
  - 각 잡에는 고유한 잡 ID (JID)가 할당되며, 프로세스 ID (PID)와 다릅니다.
  - 잡들은 그룹화되어 있는 프로세스입니다. 예를 들어 파이프(pipe) 명령어를 사용하여 여러 명령을 함께 연결할 때 프로세스들이 잡으로 그룹화됩니다.
  - 잡은 완료, 중단, 다시 시작 및 터미널에서의 작업 이동을 허용하는 작업 관리 기능을 제공합니다.

요약하면, 프로세스는 실행 중인 프로그램 인스턴스로, 일반적인 그룹화나 관리가 필요하지 않습니다. 잡은 셸에서 실행되는 프로세스의 그룹으로, 사용자가 살펴보고 관리할 수 있는 작업 관리 기능을 제공합니다.

* `jobs`
  * `jobs` list all jobs
* `fg`
  * `fg % n` Brings the current or specified job into the foreground, where n is
    the job ID
* `bg`
  * `bg % n` Places the current or specified job in the background, where n is
    the job ID
* `CTRL-Z`
  * `CTRL-Z` Stops the foreground job and places it in the background as a stopped job

## Process Management

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

## User Management

* `useradd`
  * `useradd iamslash`
* `passwd`
  * `passwd iamslash`
* `deluser`
  * `deluser iamslash`
* `pam_tally2`
  * [[LINUX] LINUX 계정 패스워드 잠김 해제](https://habiis.tistory.com/36)
  * `sudo pam_tally2 -u iamslash --reset` 계정잠김 풀어주기

## File Permissions

* `chmod`
  * `chomod -R 777 tmp`
* `chown`
  * `chown -R iamslash tmp`
* `chgrp`
  * `chgrp -R staff tmp`
* `umask`
  * [What is Umask and How To Setup Default umask Under Linux?](https://www.cyberciti.biz/tips/understanding-linux-unix-umask-value-usage.html) 
  * set user default permission for creating files, directories
  * Can set default umask adding followings to `/etc/profile`, `~/.profile`, `~/.bashrc`
  * `umask 022` 
    * create directory with permission `755 = 777 - 022`
    * create file with permission `644 = 777 - 022 - 111`. subtract additionally `111` just for files.  
  * `umask u=rwx,g=rx,o=rx` same with `umask 022`

## System Monitoring

* [Linux Performance Analysis in 60,000 Milliseconds](https://medium.com/netflix-techblog/linux-performance-analysis-in-60-000-milliseconds-accc10403c55)
  * [리눅스 서버 60초안에 상황파악하기](https://b.luavis.kr/server/linux-performance-analysis?fbclid=IwAR1VgiDybzRFhFxSpH8iBH622UArIRxcyWlEXos0wSsb4Kra6e9YMiLJP9Y)
  * `uptime`
  * `dmesg | tail`
  * `vmstat -S M 1`
  * `mpstat -P ALL 1`
  * `pidstat 1`
  * `iostat -xz 1`
  * `free -m`
    * available / Total 을 보고 scale up 판단을 한다.
  * `sar -n DEV 1`
  * `sar -n TCP,ETCP 1`
  * `top`
  * `cat /proc/meminfo`  
* `uptime`
  * `uptime`
    ```console
    $ uptime
    13:24:20 up  3:18,  0 users,  load average: 0.00, 0.01, 0.00
    ```
  * 시스템이 `13:24:20` 부터 `3:18` 동안 살아있어
  * 1 분, 5 분, 15 분의 average load 를 보여줘
  * load 는 process 들 중 run, block 인 것들의 숫자야
  * `1 분 avg load > 5 분 avg load > 15 분 avg load` 이면 점점 load 가  늘어가는 추세이기 때문에 무언가 문제가 있다고 생각할 수 있다.
* `dmesg`
  * 커널의 메시지 버퍼를 보여다오
  * `dmesg | tail`
    ```console
    $ dmesg | tail
    [1880957.563150] perl invoked oom-killer: gfp_mask=0x280da, order=0, oom_score_adj=0
    [...]
    [1880957.563400] Out of memory: Kill process 18694 (perl) score 246 or sacrifice child
    [1880957.563408] Killed process 18694 (perl) total-vm:1972392kB, anon-rss:1953348kB, file-rss:0kB
    [2320864.954447] TCP: Possible SYN flooding on port 7001. Dropping request.  Check SNMP counters.
    ``` 
    * 마지막 커널의 메시지 버퍼 10 개를 보여다오
    * 치명적인 내용이 있는지 반드시 체크해야함   
    * oom-killer(out of memory) 가 process 18694 를 kill 했다. TCP request 가 Drop 되었다.    
* `vmstat`
  * [vmstat에 대한 고찰(성능) 1편](http://egloos.zum.com/sword33/v/5976684)
    * [Vmstat에 대한 고찰(성능) 2편](http://egloos.zum.com/sword33/v/5997876)
  * [vmstat(8) - Linux man page](https://linux.die.net/man/8/vmstat)
  * virtual memory 통계 보여다오.
    | 범주 | 필드 이름 | 설명 |
    | ------ | --------- | --- |
    | procs  | r | The number of processes waiting for run time |
    | | b | The number of processes in uninterruptible sleep |
    | memory | swpd | the amount of virtual memory used in KB |
    | | free | the amout of idle memory in KB |
    | | buff | the amout of memory used as buffers in KB |
    | | cache | the amout of memory used as cache in KB |
    | | inact | the amout of inactive memory in KB |
    | | active | the amout of active memory in KB |
    | swap | si | amount of memory swapped in from disk (/s) |
    | | so | amount of memory swapped to disk (/s)                               |
    | IO | bi | blocks received from a block device (blocks/s) |
    | | bo | amount of memory swapped to disk (blocks/s) |
    | system | in | The number of interrupts per second. including the clock. |
    | | cs | The number of context switches per second. |
    | CPU | us | Time spent running non-kernel code (user time, including nice time) |
    | | sy | Time spent running kernel code (system time) |
    | | id | Time spent idle, Prior to Linux 2.5.41, this inclues IO-wait time. |
    | | wa | Time spent waiting for IO, Prior to Linux 2.5.41, inclues in idle. |
    | | st | Time stolen from a virtual machine, Prior to Linux 2.5.41, unknown. stolen time은 hypervisor가 가상 CPU를 서비스 하는 동안 실제 CPU를 차지한 시간을 이야기한다. |
  * `vmstat 1` 1 초 마다 보여다오
    ```bash
    $ vmstat 1
    procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
    r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
    6  0      0 376096  93376 788776    0    0    15    34  234   67  2  3 94  1  0
    ```
  * `vmstat -S M 1` 1 초 마다 MB 단위로 보여다오
  * `total physical memory = free + buff + cache + used`
    * buff 는 i-node 값 즉 파일들의 real address 를 cache 한다. disk seek time
      을 향상시킬 수 있다.
    * cache 는 파일의 real data 를 page size 단위로 cache 한다. disk read
      performance 를 향상시킬 수 있다.
    * free 가 부족하면 cache 에서 free 로 메모리가 옮겨갈 수도 있다. free 가
      부족하다고 꼭 메모리가 부족한 상황은 아니다.
    * `/proc/sys/vm/vfs_cache_pressure` 가 buff 와 cache 의 비율을 설정하는
      값이다. default 는 100 이다. 파일의 개수가 많아서 buff 가 중요한
      시스템이라면 이 것을 높게 설정한다.
    * `/proc/sys/vm/min_free_kbytes` 는 free 의 최소 용량이다. 이것이 낮아지면
      cache 가 높아진다. hit ratio 가 낮은 시스템인 경우 cache 가 필요 없으므로
      min_free_kbytes 를 늘려주자.
    * `/proc/sys/vm/swappiness` 는 swapping 하는 정도이다. 이것이 높으면 cache
      를 삭제하는 것보다 swapping 하는 비율이 높아진다. 이 것이 낮으면 swapping
      하는 것보다 cache 를 삭제하는 비율이 높아진다. 이 것을 0 으로 설정하면
      swapping 을 하지 않기 때문에 disk 와 memory 사이에 데이터 전송이 발생하지
      않는다. memory 가 낮고 memory 사용량이 높은 시스템의 경우 swappiness 를 0
      보다 크게 설정해야 한다.
  * `r` 이 `CPU core` 보다 크면 CPU 의 모든 core 가 일을 하고 있는 상황이다.
  * `b` 가 `CPU core` 보다 크면 disk write bottle neck 일 수 있다.
  * `wa` 가 크면 disk read bottle neck 일 수 있다.
  * `si, so` 가 0 이 아니면 메모리가 부족한 상황이다.
  * `id` 가 작으면 CPU 가 바쁜 상황이다.
  * `in` 은 인터럽트이다. 주변장치에서 CPU 에 자원을 요청하는 횟수이다. 일반
    적인 컴퓨터에서 마우스의 인터럽트가 많지만 서버의 경우는 이더넷장치와 DISK
    의 인터럽트가 많다.
  * `swapd` 는 Virtual Memory 즉 DISK 에 상주하는 VM 의 크기이다.
  * active memory are pages which have been accessed "recently", inactive memory
    are pages which have not been accessed "recently"
    * [Linux inactive memory | stackexchange](https://unix.stackexchange.com/questions/305606/linux-inactive-memory)
  * `vmstat -s` 부트이후 통계
    ```console
    $ vmstat -s
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
  * CPU 별로 CPU 의 점유율을 모니터링한다.

    | name | desc |
    | - | - |
    | CPU     | Processor number |
    | %usr    | while executing at the user level (application) |
    | %nice   | while executing at the user level with nice priority. |
    | %sys    | while executing at the system level (kernel) |
    | %iowait | percentage of time that the CPU or CPUs were idle during which the system had an outstanding disk I/O request. |
    | %irq    | percentage of time spent by the CPU or CPUs to service hardware interrupts. |
    | %soft   | percentage of time spent by the CPU or CPUs to service software interrupts. |
    | %steal  | percentage of time spent in involuntary wait by the virtual CPU or CPUs while the hypervisor was servicing another virtual processor. |
    | %guest  | percentage of time spent by the CPU or CPUs to run a virtual processor. |
    | %gnice  | percentage of time spent by the CPU or CPUs to run a niced guest. |
    | %idle   | percentage of time that the CPU or CPUs were idle and the system did not have an outstanding disk I/O request. |
  * `mpstat -P ALL 1` 1 초 마다 모든 CPU 에 대해 보여줘
* `pidstat`
  * process 별로 CPU 의 점유율을 모니터링한다.
  * `pidstat`
    ```console
    $ pidstat
    Linux 4.9.184-linuxkit (86e6c5bfb041)   11/16/19        _x86_64_        (2 CPU)

    03:59:53      UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command
    03:59:53        0         1    0.0%    0.0%    0.0%    0.0%    0.0%     1  bash
    03:59:53        0        10    0.0%    0.0%    0.0%    0.0%    0.0%     1  bash    
    ```
  * `pidstat 1` 1 초 마다 보여도
* `iotop`
  * This will display a table containing information about various processes,
    including the process ID, user, disk read and write speeds (in Kbps), swapin
    %, and I/O %. The display auto-refreshes every few seconds.
  * `sudo apt-get install iotop`
  * `sudo iotop`
* `iostat`
  * block device 별로 io 를 모니터링한다.
  * `man iostat`
  * `iostat`
    ```console
    $ iostat
    Linux 4.9.184-linuxkit (86e6c5bfb041)   11/16/19        _x86_64_        (2 CPU)

    avg-cpu:  %user   %nice %system %iowait  %steal   %idle
              2.10    0.00    2.81    1.20    0.00   93.88

    Device             tps    kB_read/s    kB_wrtn/s    kB_dscd/s    kB_read    kB_wrtn    kB_dscd
    sda              18.40        14.87        78.69         0.00     416883    2205424          0
    sr0               0.09         6.74         0.00         0.00     188798          0          0
    sr1               0.00         0.02         0.00         0.00        470          0          0
    sr2               0.05         3.52         0.00         0.00      98584          0          0    
    ``` 
  * `iostat 1` 1 초마다 보여줘
  * `iostat -xz 1`
    * `r/s, w/s, rkB/s, wkB/s` 는 각각 초당 읽기, 쓰기, kB 읽기, kB 쓰기를 의미한다.
    * `await` : The average time for the I/O in milliseconds.
    * `avgqu-sz` : The average number of requests issued to the device. 
    * `%util` : Device utilization. 
* `free`
  * ![](img/2023-11-17-20-42-12.png)
  * [리눅스 free 명령어로 메모리 상태 확인하기](https://www.whatap.io/ko/blog/37/)
  * [[Linux] Memory 확인 방법 및 종류별 설명 (Free, Buffer, Cache, Swap Memory)](https://m.blog.naver.com/PostView.nhn?blogId=yhsterran&logNo=221607492403&proxyReferer=https:%2F%2Fwww.google.com%2F)
  * `-/+ buffer/cache` 항목이 등장한다면 옛날 version 이다. `available` 이
    등장해야 새 version 이다.
  * physical memory 와 swap memory 의 상태를 알려다오.
    ```bash
    $ docker run -it --rm --name my-ubuntu ubuntu:18.04 
    # free -h
                  total        used        free      shared  buff/cache   available
    Mem:            15G        1.1G         11G        411M        3.2G         13G
    Swap:          1.0G          0B        1.0G
    # free
                  total        used        free      shared  buff/cache   available
    Mem:       16397792     1135008    11862196      421624     3400588    14644580
    Swap:       1048572           0     1048572
    ```
    * `total(16,397,792)` = `used + free + buff/cache` = `16,397,792`
    * `shared` 는 `used` 에 포함된다.
    * `free` : 누구도 점유하지 않은 Physical Memory
    * `shared` : tmpfs(메모리 파일 시스템), rmpfs 으로 사용되는 메모리.
      * 여러 프로세스에서 공유한다.
    * `buff` : For the buffer cache, used for block device I/O, saves i-node
      data (file address) to reduce DISK seek time.
    * `cache` : For the page cache and slabs, used by file systems, saves file
      data to reduce I/O. slab is a memory block managed by kernel and it is a
      part of the page cache.
    * `available` : 새로운 application 이 실행될 때 swapping 없이 사용할 수 있는 Physical memory 를 말한다. 
      * `available` : `MemFree + SReclaimable + the file LRU lists + the low watermarks in each zone` 
      * 시스템이 예측해서 계산한 것이다. 정확하다고 볼 수 없다.
      * `available` 이 부족하면 System 은 OOM (Out Of Memory) 상황이다.
    * `slab` : Kernel Object 를 저장하는 단위이다. 
      * Kernel 은 Page 보다 작은 Slab 단위로 메모리를 사용한다. 하나의 Page 에
        여러 Slab 들이 거주할 수 있다.
      * `i-node, dentry` 정보들을 캐싱한다.
  * `free -h` human readable 하게 보여줘.
  * `free -ht` total 추가해조.
  * `free -hts 5` 5초마다 갱신해서 보여줘.
* `slabtop`
  * slab 의 사용내역을 알려다오. `c` 를 누르면 CACHE SIZE 내림차순으로 보여준다.
  * [Linux Memory Slab 관리](https://lascrea.tistory.com/66)
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
  * `E` change the unit of machine memory
  * `e` change the unit of process memory
  * `c` 절대 경로 
  * `d` 갱신 시간 조정
  * `k` 프로세스에게 SIGKILL 전송
  * `r` 프로세스의 nice 를 변경하여 우선순위를 조정한다.
  * `SHIFT + m` 메모리 사용량이 큰 순서대로 정렬
  * `SHIFT + p` cpu 사용량이 큰 순서대로 정렬
  * `SHIFT + t` 실행시간이 큰 순서대로 정렬
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
  * `df -ha` 모두 알려다오
  * `df -hi` i-node 정보와 함께 알려다오.
    * Disk 가 꽉 차지 않았음에도 i-node 가 모두 사용되어 "creating `xxx.txt’: No spacse left on device" error 가 발생될 수 있다.
  * `df -hT` 파일 시스템타입과 함께 알려다오
  * `df -ht ext3` ext3 포함해서 알려줘
  * `df -hx ext3` ext3 빼고 알려줘
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
  * `netstat -anv | grep 8080` 8080 포트 사용하는 process id 는?
    * `ps aux | grep xxxx`
    * [How to Find the Process Listening to Port on Mac OS X](https://www.btaz.com/mac-os-x/find-the-process-listening-to-port-on-mac-os-x/)
    * [3 Ways to Find Out Which Process Listening on a Particular Port](https://www.tecmint.com/find-out-which-process-listening-on-a-particular-port/)
  * `netstat -nat | awk '{print $6}' | sort | uniq -c | sort -n` list tcp connected endpoints
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
  * [네트워크 인터페이스 다루기](https://thebook.io/006718/part01/ch03/06/01/)
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
  * 열린 파일들을 보여다오.
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
  * 하드웨어 정보들을 알려다오.
  * `sudo lshw`
  * `sudo lshw -short` 짧게 부탁해
  * `sudo lshw -short -class memory` 메모리분야만 부탁해
  * `sudo lshw -class processor` `sudo lshw -short -class disk` `sudo lshw -class network`
  * `sudo lshw -businfo` pci, usb, scsi, ide자치들의 주소를 부탁해
  * `sudo lshw -html > hardware.html` `sudo lshw -xml > hardware.xml`
* `who`
  * 로그인한 녀석들 보여다오.
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

## Logs

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
  * `grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt` PCRE (perl compatible regular expression)
    * file.txt 에서 `000-000-0000` 혹은 `(000) 000-0000` 만 보여다오
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
  * 구분자 `\n` 을 이용해서 argument list 를 구성하고 command 한개에 argument
    하나씩 대응해서 실행하자
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
  * A versatile programming language in itself, awk scans lines for patterns and
    can manipulate the output. It's commonly used for data extraction and
    reporting.
  * [리눅스 awk 사용법](https://zzsza.github.io/development/2017/12/20/linux-6/)
  * `cat a.txt | awk '{print $4}'`
    * a.txt 를 읽어서 매 줄마다 4 번 째 컬럼만 출력하라
  * `cat a.txt | awk -F, '{print $2, $4}`
    * a.txt 를 읽어서 매 줄마다 `,` 를 구분자로 해서 2, 4 번 째 컬럼만 출력하라. 
* `uniq`
  * This utility filters out or reports repeated lines in a file. It's often
    used in conjunction with sort to count unique lines.
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
  * As the name implies, sort can order the lines of a file in a particular
    sequence, either numerically or alphabetically, and has various options to
    customize the output.
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
* `lnav`
  * [lnav](http://lnav.org/)
  * 괜찮은 text syslog viewer
* `journalctl`
  * [[Linux] journalctl을 이용해 systemd 로그 확인하기](https://twpower.github.io/171-journalctl-usage-and-examples)
  * Specific to systemd, journalctl queries and displays entries from the
    systemd journal, which is a system service for collecting and storing log
    data.
  * systemd log 확인
  * `journalctl -u <systemd-unit-name>`

## Text

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
* [sed](/sed/README.md)
  * `sed -e 's/regex/REGEXP/g' a.txt`
    * a.txt 를 읽어서 모든 regex 를 REGEXP 로 바꿔라
  * `sed -e 's/regex/REGEXP/g' a.txt > b.txt`
  * `sed -e 's/regex/REGEXP/g' -i a.txt`
    * a.txt 를 읽어서 모든 regex 를 REGEXP 로 바꿔서 a.txt 에 저장하라
* `jq`
  * [jq](/jq/README.md)
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
  * print formatted string.
  * `printf "%s\n" "hello printf"`
  * `printf "%s\n" "hllo printf" "in" "bash script"` 각 문자열에 적용해서 출력해라.
* `nl`
  * 줄번호 보여줘라.
  * `nl a.txt`
  * `nl a.txt > nla.txt`
  * `nl -i5 a.txt` 줄번호는 5씩 더해서 보여다오
  * `nl -w1 a.txt` 줄번호 열위치를 다르게 해서 보여다오
  * `nl -bpA a.txt` A로 시작하는 문자열만 줄번호 적용해라
  * `nl -nln a.txt` `nl -nrn a.txt` `nl -nrz a.txt`
* `seq`
  * sequence number 출력해 다오.
  * `seq 10`
  * `seq 35 45`
  * `seq -f "%02g/03/2016" 31`
  * `seq 10 -1 1` 거꾸로 보여줘 
* `sha1sum`
  * SHA-1 message digest알려줄래?
  * `sha1sum a.txt`
* `md5sum`

## Debugging

* `gdb`
* `ldd`
  * shared library 의존성을 알려다오.
  * `ldd execv`
  * `ldd a.so`
  * `ldd -v a.so`
  * `ldd -u func` unused direct dependencies
  * `ldd /bin/ls` ldd wants absolute path
* `nm`
  * symbol table을 알려다오.
  * `nm  -A ./*.o | grep func` func를 포함한 심볼들을 알려주라
  * `nm -u a.out` 사용하지 않거나 실행시간에 share libary에 연결되는 심볼들을 알려주라
  * `nm -n a.out` 모든 심볼들을 알려주라
  * `nm -S a.out | grep abc` abc를 포함한 심볼을 크기와 함께 알려다오
  * `nm -D a.out` dynamic 심볼좀 알려주라
  * `nm -u -f posix a.out` bsd형식 말고 posix형식으로 알려다오
  * `nm -g a.out` external 심볼만 알려다오
  * `nm -g --size-sort a.out` 실볼을 크기 정렬해서 알려다오
* `strace`
  * system call 과 signal 을 모니터링 해주라. 소스는 없는데 디버깅 하고 싶을때
    유용하다.
  * `strace ls`
  * `strace -c ls` Show ltrace with counts
  * `strace -c ls /home`
  * `strace -e open ls` system call중 open만 보여주라
  * `strace -e trace-open,read ls /home` system call중 open, read보여주라
  * `strace -o a.txt ls` `cat a.txt`
  * `ps -C firefox-bin` `sudo strace -p 1725 -o firefox_trace.txt` `tail -f firefox_trace.txt`
  * `strace -p 1725 -o output.txt`
  * `strace -t -e open ls /home` 절대 시간으로 보여다오
  * `strace -r ls` 상대 시간으로 보여다오
* `ltrace`
  * library call 좀 보여다오
  * `ltrace ls`
  * `ltrace -c ls` Show ltrace with counts
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

## Compressions

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

## Editors

* `vi`
* [vim](/vim/README.md)
* `emacs`
* `nano`
* `ed`

## Files

* `file`
  * `file a.txt` 파일종류좀 알려줘
  * `sudo file -s /dev/xvdf/` 특별한 파일 즉 디스크좀 알려줘. 포맷되었는지 확인할 수 있다.
* [rsync](/rsync/README.md)
  * [rsync @ man](https://linux.die.net/man/1/rsync)
  * [sync 사용법 - data backup 포함](https://www.lesstif.com/pages/viewpage.action?pageId=12943658)
    * [Rsync](https://www.lesstif.com/display/1STB/rsync)
  * `rsync -n -avrc /abc/home/sample1/* server2:/abc/home/sample2/` rsync dry-run
  * `rsync -avz --progress -e 'ssh -p 10022' iamslash@example.com:/home/hello /home/world`
    * `--progress` option: 진행상황 확인 
  * `rsync -am --include='config.xml' --include='*/' --prune-empty-dirs --exclude='*' $JENKINS_HOME/jobs/ $BUILD_ID/jobs/` rsync from old Jenkins job config files to new Jenkins job config files
    * `--include=<PATTERN>` include `<PATTERN>`
    * `--exclude=<PATTERN>` exclude `<PATTERN>`
    * `-m, --prune-empty-dirs` Skip empty dirs for sync
* `fuser`
  * `fuser -m /etc/sshd/sshd_config` sshd_config 파일을 사용하는 프로세스의 PID 를 확인
  * `fuser -k /usr/sbin/sshd` sshd 데몬을 사용하고 있는 프로세스를 모두 KILL
  * `fuser -v /usr/sbin/sshd` 특정 프로세스(또는 데몬파일)가 실행 중에 사용한 user, pid, access 정보 

## Daemon Management

* `cron`
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
* `systemd`
  * [systemd](/systemd/README.md)

## Disk

* [Ubuntu 14.04에 새로운 하드디스크 추가 및 포맷후 자동 마운트 설정](http://reachlab-kr.github.io/linux/2015/10/03/Ubuntu-fstab.html)
* [Amazon EBS 볼륨을 Linux에서 사용할 수 있도록 만들기](https://docs.aws.amazon.com/ko_kr/AWSEC2/latest/UserGuide/ebs-using-volumes.html)

----

* `df`
  * `df -hT` show disk free by filesystem with Type.
    ```bash
    $ df -hT
    Filesystem      Size  Used Avail Use% Mounted on
    overlay          59G  7.3G   49G  14% /
    tmpfs            64M     0   64M   0% /dev
    tmpfs          1000M     0 1000M   0% /sys/fs/cgroup
    shm              64M     0   64M   0% /dev/shm
    /dev/sda1        59G  7.3G   49G  14% /etc/hosts
    tmpfs          1000M     0 1000M   0% /proc/acpi
    tmpfs          1000M     0 1000M   0% /sys/firmware`
    ```
  * `df -hi` show disk inodes.
* `resize2fs`
  * EBS volume 을 추가하고 EC2 instace 에서 디스크 크기를 증가한다.
  * `resize2fs /dev/sdb` resize `/dev/sdb`
* `fdisk`
  * `fdisk -l` show hard disk partitions
  * `fdisk /dev/sdb` create partition
* `mkfs`
  * file system 을 생성한다.
  * `mkfs -t ext4 /dev/sdb1` format `/dev/sdb1` with ext4 type
* `blkid` 
  * `blkid` list properties of block devices including UUID for mount
* `mount`
  * `mount` show all mounts
  * `mount -t ext4 /dev/sdb1 /mnt/dir1`
  * `mount -a`  mount all filesystems mentioned in fstab
  * `vim /etc/fstab` add mount information and mount automatically.
    * Linux mount with reading `/etc/fstab` everytime it boots.
* `umount`
  * `umount /mnt/dir1` unmount `/mnt/dir1`
* `lsblk`
  * EBS volume 을 추가하고 EC2 instane 에서 확인할 수 있다.
  * `lsblk` list block devices

## Network

* [route](/route/README.md)
  * [라우팅 테이블 다루기](https://thebook.io/006718/part01/ch03/06/02/)
  * [Ubnutu route 설정](https://xmlangel.github.io/ubuntu-route/)
  * `route` route 목록을 보여다오
    * `netstat -nr` 과 같다. macos 에서 가능.
  * `sudo route add default gw 192.168.0.1` default 목적지에 Gateway 를 추가해 다오
* `ufw`
  * [ufw firewall deny outgoing but allow browser](https://askubuntu.com/questions/1005312/ufw-firewall-deny-outgoing-but-allow-browser)
  * `sudo ufw status verbose` Show lists of rules 
  * `sudo ufw allow out to any port 443` allow outbound 443
    * `sudo ufw reload` Should reload ufw
* `socat`
  * socket + cat
  * 주로 특정 socket 으로 수집된 데이터를 다른 곳으로 relay 할 때 사용한다.
  * [socat 을 이용해 소켓 조작하기](https://blog.naver.com/parkjy76/220630685718)
  * [socat(1) - Linux man page](https://linux.die.net/man/1/socat)
  * `socat -v tcp-listen : 8080 system : cat` 8080 port tp listen, cat standard-out
    * `telnet localhost 8080`
  * `socat -v tcp-listen : 8080 fork system : cat` Fork every connecion.
  * `socat -v tcp-listen : 8080 fork system : cat` launch server
    * `socat -v unix-listen : proxysocket, reuseaddr, fork tcp-connect : localhost : 8080` launch proxy
      * `socat STDIN unix-connect : proxysocket` launch client
  * `socat -v TCP-LISTEN:80,fork TCP:iamslash.com:8000` foward local:80 to iamslash.com:8080

## Automation

* [expect](/expect/README.md)

## Oneline Commands

* 파일을 읽고 "hello" 혹은 "world" 를 가진 라인들을 고른다. 그리고 적절히 필터링 한다.

```bash
$ cat a.csv | \
      grep -e hello \
       -e world \
       | awk '{print $1, $12}' | awk -F, '{print $7}' | grep foo | sed 's/"//g' | sort | uniq
```

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

## Tips

* execute muli-line commands on bash, zsh
  * `C-x C-e` 
  * [셸에서 여러줄의 명령어를 에디터로 편집하고 실행하기 @ 44bits](https://www.44bits.io/ko/post/editing-multiline-command-on-shell)

# Exit Codes

> * [Appendix E. Exit Codes With Special Meanings](https://tldp.org/LDP/abs/html/exitcodes.html)

```
137 = 128 + 9  (SIG KILL)
143 = 128 + 15 (SIG TERM)
```

# Performance Monitoring

> [awesome performance testing | github](https://github.com/andriisoldatenko/awesome-performance-testing)

* [stress](https://linux.die.net/man/1/stress)
  * `stress -c 4` Use 4 cores 100%
  * `stress -vm 2 –vm-bytes 2048m` Use 2 process with 2048 memory
  * `stress –hdd 3 -hdd-bytes 1034m` Use 3 hdds with 1034MB size
  * `stress --cpu 8 --io 4 --vm 2 --vm-bytes 128M --timeout 10s`
  * `info stress`
* [stress-ng](https://kernel.ubuntu.com/~cking/stress-ng/)
  * [src @ github](https://github.com/ColinIanKing/stress-ng)
  * `ng` means new generation???
  * `stress-ng --cpu 4 --vm 2 --hdd 1 --fork 8 --switch 4 --timeout 5m --metrics-brief` 
  * `stress-ng --vm 32 --vm-bytes 64M --vm-stride 1K --vm-populate --page-in --metrics-brief --times --timeout 60s`
  * `` 
* [k6](https://github.com/loadimpact/k6)

# Security

## root 소유의 setuid, setgid 파일 검색 후 퍼미션 조정하기

owner 가 root 인 파일들을 생각해보자. setuid 가 설정되어 있으면 실행 되었을 때 EUID 가 root 로 변경된다. 매우 위험하다. 그러한 파일들을 찾아서 위험해 보인다면 권한을 변경해 준다.

```bash
find / -user root -perm 4000 -print
```

# System Monitoring

## Load Average

[System Load](/linuxkernel/README.md#system-load)

## Swapin, Swapout

[An introduction to swap space on Linux systems](https://opensource.com/article/18/9/swap-space-linux-systems)

----

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

## Memory

* [How to calculate system memory usage from /proc/meminfo (like htop) @ stackoverflow](https://stackoverflow.com/questions/41224738/how-to-calculate-system-memory-usage-from-proc-meminfo-like-htop)
* [E.2.18. /PROC/MEMINFO](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/deployment_guide/s2-proc-meminfo)

----

![](img/2023-11-17-20-42-12.png)

`/proc` directory 는 process 들의 정보를 담고 있다. [proc | kernel](https://www.kernel.org/doc/Documentation/filesystems/proc.txt) 참고. [The /proc Filesystem | kernel](https://www.kernel.org/doc/html/latest/filesystems/proc.html?highlight=meminfo) 은 좀더 보기 편하다.

특히 `/proc/meminfo` 는 Linux Kernel 의 memory 정보가 저장되어 있다. `free` 를 포함한
대부분의 메모리 조회 application 들은 `/proc/meminfo` 의 내용을 참고한다.

```bash
$ docker run --rm -it --name my-ubuntu ubuntu:18.04

$ cat /proc/meminfo
MemTotal:        8152552 kB
MemFree:         2532460 kB
MemAvailable:    5761976 kB
Buffers:          425932 kB
Cached:          3307452 kB
SwapCached:          144 kB
Active:          1099632 kB
Inactive:        4194012 kB
Active(anon):      75120 kB
Inactive(anon):  1841580 kB
Active(file):    1024512 kB
Inactive(file):  2352432 kB
Unevictable:           0 kB
Mlocked:               0 kB
SwapTotal:       1048572 kB
SwapFree:        1047536 kB
Dirty:                12 kB
Writeback:             0 kB
AnonPages:       1557008 kB
Mapped:           252812 kB
Shmem:            412440 kB
KReclaimable:     188784 kB
Slab:             257616 kB
SReclaimable:     188784 kB
SUnreclaim:        68832 kB
KernelStack:       11104 kB
PageTables:         8128 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:     5124848 kB
Committed_AS:    6654600 kB
VmallocTotal:   34359738367 kB
VmallocUsed:       13744 kB
VmallocChunk:          0 kB
Percpu:             9984 kB
AnonHugePages:      2048 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
FileHugePages:         0 kB
FilePmdMapped:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:      229376 kB
DirectMap2M:     8159232 kB
DirectMap1G:     2097152 kB
```

[/fs/proc/meminfo.c](https://elixir.bootlin.com/linux/v4.15/source/fs/proc/meminfo.c#L46) 를 참고하면 주요지표들이 어떻게 계산되는지 알 수 있다.

다음은 주요지표를 요약한 것이다.

```
    MemTotal: Total usable ram (i.e. physical ram minus a few reserved
              bits and the kernel binary code)
     MemFree: The sum of LowFree+HighFree
MemAvailable: An estimate of how much memory is available for starting new
              applications, without swapping. Calculated from MemFree,
              SReclaimable, the size of the file LRU lists, and the low
              watermarks in each zone.
              The estimate takes into account that the system needs some
              page cache to function well, and that not all reclaimable
              slab will be reclaimable, due to items being in use. The
              impact of those factors will vary from system to system.
     Buffers: Relatively temporary storage for raw disk blocks
              shouldn't get tremendously large (20MB or so)
      Cached: in-memory cache for files read from the disk (the
              pagecache).  Doesn't include SwapCached
  SwapCached: Memory that once was swapped out, is swapped back in but
              still also is in the swapfile (if memory is needed it
              doesn't need to be swapped out AGAIN because it is already
              in the swapfile. This saves I/O)
      Active: Memory that has been used more recently and usually not
              reclaimed unless absolutely necessary.
    Inactive: Memory which has been less recently used.  It is more
              eligible to be reclaimed for other purposes
   HighTotal:
    HighFree: Highmem is all memory above ~860MB of physical memory
              Highmem areas are for use by userspace programs, or
              for the pagecache.  The kernel must use tricks to access
              this memory, making it slower to access than lowmem.
    LowTotal:
     LowFree: Lowmem is memory which can be used for everything that
              highmem can be used for, but it is also available for the
              kernel's use for its own data structures.  Among many
              other things, it is where everything from the Slab is
              allocated.  Bad things happen when you're out of lowmem.
   SwapTotal: total amount of swap space available
    SwapFree: Memory which has been evicted from RAM, and is temporarily
              on the disk
       Dirty: Memory which is waiting to get written back to the disk
   Writeback: Memory which is actively being written back to the disk
   AnonPages: Non-file backed pages mapped into userspace page tables
HardwareCorrupted: The amount of RAM/memory in KB, the kernel identifies as
	      corrupted.
AnonHugePages: Non-file backed huge pages mapped into userspace page tables
      Mapped: files which have been mmaped, such as libraries
       Shmem: Total memory used by shared memory (shmem) and tmpfs
ShmemHugePages: Memory used by shared memory (shmem) and tmpfs allocated
              with huge pages
ShmemPmdMapped: Shared memory mapped into userspace with huge pages
KReclaimable: Kernel allocations that the kernel will attempt to reclaim
              under memory pressure. Includes SReclaimable (below), and other
              direct allocations with a shrinker.
        Slab: in-kernel data structures cache
SReclaimable: Part of Slab, that might be reclaimed, such as caches
  SUnreclaim: Part of Slab, that cannot be reclaimed on memory pressure              
```

* **MemAvailable** : 새로운 Application 이 실행될 때 Swapping 없이 사용할 수 있는 Physical Memory 를 말한다. **MemFree**, **SReclaimable**, **the file LRU lists**, **the low watermarks in each zone** 으로 구성된다.
  * [/mm/page_alloc.c](https://elixir.bootlin.com/linux/v4.15/source/mm/page_alloc.c#L4564) 를 참고하면 **MemAvailable** 이 어떻게 계산되는지 알 수 있다.
* Inactive(anon), Inactive(file), SReclaimable 은 사용된지 오래된 메모리들의 모임이다. System 이 Memory Pressure 상황이라면 Physical Memory 에서 날아갈 수 있다.
  * Inactive(anon) : Anonymous Memory 중 사용된지 오래된 메모리이다. Stack, Heap 등을 말한다. Swap Out 의 대상이다.
  * Inactive(file) : Page cache 중 사용된지 오래된 메모리이다. Swap Out 의 대상은 아니다.
  * SReclaimable : Slab 중 다시 할당하는 것이 가능한 것들이다. Swap Out 의 대상일까???

[htop](https://htop.dev/) 은 개선된 top 이다. top 보다 직관적이다. [htop](https://htop.dev/) 의 주요 지표들을 해석해 보자.

![](img/htop.png)

>* [htop/linux/LinuxProcessList.c](https://github.com/hishamhm/htop/blob/8af4d9f453ffa2209e486418811f7652822951c6/linux/LinuxProcessList.c#L802-L833) 
>* [htop/linux/Platform.c](https://github.com/hishamhm/htop/blob/1f3d85b6174f690a7e354bbadac19404d5e75e78/linux/Platform.c#L198-L208)

* Total used memory = MemTotal - MemFree
* Non cache/buffer used memory (green) = Total used memory - (Buffers + Cached memory)
* Buffers (blue) = Buffers
* Cached memory (yellow) = Cached + SReclaimable - Shmem
* Swap = SwapTotal - SwapFree

# Network Kernel Parameters

* `net.ipv4.tcp_tw_reuse`
  * [tcp_tw_reuse and tcp_tw_recycle](https://brunch.co.kr/@alden/3)
  * time wait 상태의 socket 을 재사용할 수 있게 해준다.
* `net.core.somaxconn`
  * [리눅스 서버의 TCP 네트워크 성능을 결정짓는 커널 파라미터 이야기 - 2편](https://meetup.toast.com/posts/54)
  * accept() 을 기다리는 ESTABLISHED 상태의 소켓 (connection completed) 을 위한 queue
* `net.ipv4.tcp_max_syn_backlog`
  * SYN_RECEIVED 상태의 소켓(connection incompleted)을 위한 queue
* `net.core.netdev_max_backlog`
  * 각 네트워크 장치 별로 커널이 처리하도록 쌓아두는 queue 의 크기. 커널의 패킷 처리 속도가 상대적으로 느리다면 queue 에 패킷이 쌓일 것이고 queue 에 추가되지 못한 패킷들은 버려진다. 적당히 설정해야 함.

```bash
$ echo "net.core.netdev_max_backlog = 65536" >> /etc/sysctl.conf
$ echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
$ echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
$ echo "net.ipv4.tcp_tw_reuse = 1" >> /etc/sysctl.conf
```

# File Kernel Parameters

* [리눅스 최대 열수 있는 파일 갯수 수정하기(Linux increase the max open files per user)](https://www.lesstif.com/lpt/linux-increase-the-max-open-files-per-user-48103542.html)
  * [Linux Increase The Maximum Number Of Open Files / File Descriptors (FD)](https://www.cyberciti.biz/faq/linux-increase-the-maximum-number-of-open-files/)
* [Docker & Kernel Configuration @ m3](https://m3db.github.io/m3/operational_guide/kernel_configuration/)
* [Java, max user processes, open files](https://woowabros.github.io/experience/2018/04/17/linux-maxuserprocess-openfiles.html)

-----

```bash
# Read parameters 
$ sudo sysctl -n fs.file-max
$ sudo sysctl -n fs.nr_open

# Read
$ ulimit -a
$ ulimit -Sn 3000000

# Write parameters to /etc/sysctl.conf
$ sudo echo "fs.file-max=3000000" >> /etc/sysctl.conf
$ sudo echo "fs.nr_open=3000000" >> /etc/sysctl.conf
# Load parameters
$ sudo sysctl -p

# Write parameters to /etc/security/limits.conf
$ vim /etc/security/limits.conf
ubuntu	soft	nproc 		10000
ubuntu	hard	nproc		10000
ubuntu	soft 	nofile		3000000
ubuntu	hard	nofile		3000000

# Check "session required pam_limits.so" in /etc/pam.d/login
$ sudo vim /etc/pam.d/login
session required pam_limits.so

# Check other limits in /etc/security/limits.d/*.conf
$ sudo vim /etc/security/limits.d/*.conf

# Check again after log-out and log-in again
$ ulimit -a
```

max process id 를 바꾸어 줘야할 수도 있다. [Maximum PID in Linux @ stackoverflow](https://stackoverflow.com/questions/6294133/maximum-pid-in-linux)

```bash
# We need to increase **pid_max** to launch many processes or 
# you will encounter `fork: retry: No child processes` error !!!
$ sudo echo 4194304 > /proc/sys/kernel/pid_max
```

# Cgroup

> * [cgroups](https://hwwwi.tistory.com/12)
> * [CGROUPS | kernel](https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt)

프로세스들을 하나의 그룹으로 묶는다. 그리고 cpu, memory, network, device, I/O
등등의 자원을 그룹별로 제한할 수 있다. [Docker](/docker/README.md) 와 같은
container 기술의 기반이 된다. Docker Container 가 하나 실행되면 control group 이
생성될 것 같다.

cgroup 은 다음과 같은 방법으로 설정할 수 있다.

* `/sys/fs/cgroup` 을 직접 편집
* `cgmanager`
* `cgroup-tools`

다음은 `cgroup-tools` 를 이용하여 control group 을 만들고 cpu 를 제한하는 예이다.

```bash
$ sudo cgcreate -a hwi -g cpu:mycgroup
$ ls -al /sys/fs/cgroup/cpu/ | grep mycgroup
drwxr-xr-x  2 hwi root   0 10월 20 18:00 mycgroup
```

cpu 를 제한하기 위해 mycgroup 이라는 control group 을 생성했다. 소유자는 hwi
이다. control cgroup 을 생성하면 제한하고자 하는 subsystem 의 디렉토리에 cgroup
directory 가 만들어 진다. 

이제 테스트 해보자.

```bash
$ stress -c 1
$ top
$ sudo cgset -r cpu.cfs_quota_us=30000 mycgroup
$ sudo cgexec -g cpu:mycgroup stress -c 1
```

이제 cgroup 을 삭제해 보자.

```
$ sudo cgdelete cpu:mycgroup
$ ls -al /sys/fs/cgroup/cpu/ | grep mycgroup
```

# Slab

> [slabtop | tistory](https://ssup2.github.io/command_tool/slabtop/)

Slab 은 Kernel object 를 caching 한다. 다수의 slab 으로 구성된다. 하나의 slab 은
page size (`4K`) 와 같다. 하나의 slab 은 여러 kernel object 들로 구성된다. 

다음과 같은 수식이 성립한다.

* `4KB * SLABS = CACHE SIZE`
* `OBJ/SLAB * OBJ SIZE < 4KB`

다음과 같이 slab 을 flushing 할 수도 있다.

```bash
$ echo 1 > /proc/sys/vm/drop_caches # Page cache
$ echo 2 > /proc/sys/vm/drop_caches # inode, dentry cache
$ echo 3 > /proc/sys/vm/drop_caches # Page, inode, dentry cache
```
