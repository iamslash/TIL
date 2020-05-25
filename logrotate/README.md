# Materials

* [Logrotate를 이용한 로그파일 관리](https://blog.naver.com/ncloud24/220942273629)
* [logrotate를 활용한 로그 관리 (compress, lotate)](https://blueskai.tistory.com/101)
* [logrotate(8) - Linux man page](https://linux.die.net/man/8/logrotate)

# Basic

## How to work logrotate

* `cron.daily` will call `/usr/sbin/lorotate` with `/etc/logrotate.conf`, `/etc/logrotate.d/*`.

## Set up configs

* edit logrotate config
  * `vim /etc/logrotate.conf`

    ```bash
    ...
    # packages drop log rotation information into this directory
    include /etc/logrotate.d
    ...
    ```

* edit custom logrotate config
  * `vim /etc/logrotate.d/echotime.conf`

    ```bash
    /iamslash/logs/*/*.log {
        hourly
        rotate 72
        dateext
        dateformat -%Y%m%d-%s
        compress
        delaycompress
        copytruncate
        missingok
        notifempty
    }
    ```

* run

```bash
$ /usr/sbin/logrotate /etc/logrotate.conf
```

* register to cron hourly
  * `vim /etc/cron.hourly/logrotate`

    ```bash
    #!/bin/sh
    ...
    test -x /usr/sbin/logrotate || exit 0
    /usr/sbin/logrotate /etc/logrotate.conf
    ```

## 