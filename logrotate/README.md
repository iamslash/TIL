# Basic

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
