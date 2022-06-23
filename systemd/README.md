- [Materials](#materials)
- [Usual Commands](#usual-commands)
- [Simple Service](#simple-service)
- [Advanced](#advanced)
  - [Restart Periodically](#restart-periodically)

----

# Materials

* [Creating systemd Service Files @ youtube](https://www.youtube.com/watch?v=fYQBvjYQ63U)
* `man system.service`
* [systemd @ wikipedia](https://en.wikipedia.org/wiki/Systemd)
* [systemd unit 등록 관련 옵션 정리](https://fmd1225.tistory.com/93)

# Usual Commands

```
$ systemctl show httpd
```

# Simple Service

* `$ vim ~/echotime.sh`

```bash
#!/bin/bash
while true
do
  echo The current time is $(date)
  sleep 1
done
```

* `$ sudo vim /etc/systemd/system/echotime.service`

```s
[Service]
ExecStart=/home/iamslash/echotime.sh
```

* run `echotime.service`

```bash
$ sudo systemctl start echotime.service
$ sudo systemctl status echotime.service
# redirect standard output to syslog
$ tail /var/log/syslog
$ sudo journalctl -u echotime
# same with tail -f
$ sudo journalctl -u echotime -f
$ sudo systemctl stop echotime
```

* adjust more `$ sudo vim /etc/systemd/system/echotime.service`

```
[Unit]
Description=Echo time service
After=network.target

[Service]
ExecStart=/home/iamslash/echotime.sh
Restart=always
WorkingDirectory=/home/iamslash
User=iamslash
Group=iamslash
Environment=GOPATH=/home/iamslash/my/go

[Install]
WantedBy=multi-user.target
```

* register `echotime.service`

```bash
$ sudo systemctl enable echotime.service
$ sudo systemctl daemon-reload
$ sudo systemctl restart echotime
$ sudo systemctl status echotime
$ sudo systemctl edit echotime --full
```

# Advanced

## Restart Periodically

* [](https://stackoverflow.com/questions/31055194/how-can-i-configure-a-systemd-service-to-restart-periodically)

----

* `Type=notify`
* `Restart=always`
* `WatchdogSec=xx`, where xx is the time period in second you want to restart your service.

```
[Unit]
.
.
[Service]
Type=notify
.
.
WatchdogSec=10
Restart=always
.
.
[Install]
WantedBy= ....
```
