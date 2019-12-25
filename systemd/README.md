# Materials

* [Creating systemd Service Files @ youtube](https://www.youtube.com/watch?v=fYQBvjYQ63U)
* `man system.service`

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