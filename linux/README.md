# Abstract

linux를 활용할 때 필요한 지식들을 정리한다.
unix 계열 os는 모두 해당된다.

# Directories

-------
|DIRECTORY	| DESCRIPTION|
/etc/passwd	Contains local Linux users.
/etc/shadow	Contains local account password hashes.
/etc/group	Contains local account groups.
/etc/init.d/	Contains service init script ? worth a look to see whats installed.
/etc/hostname	System hostname.
/etc/network/interfaces	Network interfaces.
/etc/resolv.conf	System DNS servers.
/etc/profile	System environment variables.
~/.ssh/	SSH keys.
~/.bash_history	Users bash history log.
/var/log/	Linux system log files are typically stored here.
/var/adm/	UNIX system log files are typically stored here.
/var/log/apache2/access.log
/var/log/httpd/access.log

Apache access log file typical path.
/etc/fstab	File system mounts.

# Permissions

# Commands

application commands와 bash builtin commands등이 있다.  상황별로
유용한 commands를 정리한다. bash builtin commands의 경우 `/usr/bin/`
에 application commands으로 존재한다. 다음은 macosx에서
`/usr/bin/ulimit`의 내용이다. `${0##*/}` 과 `${1+"$@"}`는 parameter
expansion을 의미한다.


```bash
builtin `echo ${0##*/} | tr \[:upper:] \[:lower:]` ${1+"$@"}
```

## eshell
