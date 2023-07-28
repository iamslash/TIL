- [Abstract](#abstract)
- [Materials](#materials)
- [Alternatives](#alternatives)
- [Basics](#basics)
  - [Restart nodes](#restart-nodes)

-----

# Abstract

programmed dialogue with interactive programs

# Materials

* [expect](https://core.tcl-lang.org/expect/index)
  * [src @ sourceforge](https://sourceforge.net/projects/expect/)
  * [src @ github](https://github.com/aeruder/expect)
* [expect(1) - Linux man page](https://linux.die.net/man/1/expect)
* [expect 를 이용한 자동화 프로그래밍](https://www.joinc.co.kr/w/Site/JPerl/expect)
* [Hak5 - Automate Everything, Using Expect, Hak5 1023.1](https://www.youtube.com/watch?v=dlwqyMW5H5I)

# Alternatives

* [empty by sh @ sourceforge](http://empty.sourceforge.net/)
* [goexpect by go @ github](https://github.com/google/goexpect)
* [peexpect by python @ github](https://github.com/pexpect/pexpect)

# Basics

## Restart nodes

ssh to machines and restart docker containers. password should be escaped because it can include `$`.

```sh
#!/bin/bash

# ./restartmachines.sh dev iamslash '<password>'
export profile=$1
export username=$2
export password=$3
export bastion="gw-dev.iamslash.com"
export hostlist="
10.111.111.11
10.111.111.11
"
if [ "$profile" == "prod" ]; then

  bastion="gw-prod.iamslash.com"
  hostlist="
10.111.112.11
10.111.112.11
"
fi

# escape password
password=${password/\$/\\\$}

function restart_docker {
    ip="$1"
    /usr/bin/expect -c "
proc login_ssh {} {
  expect {
    \"Are you sure you want to continue connecting\" {
      send -- \"yes\n\"
      exp_continue
    }

    \"Please use your credential!\" {
      expect {
        -re \".*?assword:\" {
          send -- \"$password\n\"
        }
      }
    }
  }
  expect \"@\"
  send -- \"\n\"  
}

proc setup_sh {} {
  send -- \"unset TMOUT\n\"
  send -- \"alias ls='ls --color=auto'\n\"
  send -- \"export LANG=en_US.UTF-8\n\"
  send -- \"cd ~\n\"
  send -- \"sudo su iamslash\n\"
}

proc restart_docker_compose {} {
  send -- \"cd /data/service/app\n\"
  send -- \"docker ps\n\"
  send -- \"sleep 1\n\"
  send -- \"exit\n\"
  send -- \"sleep 1\n\"
  send -- \"exit\n\"
}

send -- \"export LANG=en_US.UTF-8\n\"
spawn ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $username@$bastion
login_ssh

send -- \"export LANG=en_US.UTF-8\n\"
send -- \"ec2-ssh.sh $ip\n\"  
login_ssh

setup_sh
restart_docker_compose

interact
"
}

for hostname in $hostlist; do
    # echo "$hostname"
    restart_docker "$hostname"
done
```
