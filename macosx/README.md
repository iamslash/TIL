- [Basic](#basic)
  - [Open Files Limit](#open-files-limit)
  - [lsof](#lsof)

----

# Basic

## Open Files Limit

> [How to Change Open Files Limit on OS X and macOS](https://gist.github.com/tombigel/d503800a282fcadbee14b537735d202c)

----

확인하는 방법

```
launchctl limit maxfiles
```

수정하는 방법. rebooting 해야함.

```bash
#!/bin/sh

# These are the original gist links, linking to my gists now.
# curl -O https://gist.githubusercontent.com/a2ikm/761c2ab02b7b3935679e55af5d81786a/raw/ab644cb92f216c019a2f032bbf25e258b01d87f9/limit.maxfiles.plist
# curl -O https://gist.githubusercontent.com/a2ikm/761c2ab02b7b3935679e55af5d81786a/raw/ab644cb92f216c019a2f032bbf25e258b01d87f9/limit.maxproc.plist

curl -O https://gist.githubusercontent.com/tombigel/d503800a282fcadbee14b537735d202c/raw/ed73cacf82906fdde59976a0c8248cce8b44f906/limit.maxfiles.plist
curl -O https://gist.githubusercontent.com/tombigel/d503800a282fcadbee14b537735d202c/raw/ed73cacf82906fdde59976a0c8248cce8b44f906/limit.maxproc.plist

sudo mv limit.maxfiles.plist /Library/LaunchDaemons
sudo mv limit.maxproc.plist /Library/LaunchDaemons

sudo chown root:wheel /Library/LaunchDaemons/limit.maxfiles.plist
sudo chown root:wheel /Library/LaunchDaemons/limit.maxproc.plist

sudo launchctl load -w /Library/LaunchDaemons/limit.maxfiles.plist
sudo launchctl load -w /Library/LaunchDaemons/limit.maxproc.plist
```

## lsof

* [Who is listening on a given TCP port on Mac OS X? @ stackoverflow](https://stackoverflow.com/questions/4421633/who-is-listening-on-a-given-tcp-port-on-mac-os-x)

```bash
# List tcp
$ sudo lsof -itcp
# Without port no convertion to name 
$ sudo lsof -itcp -P
# List tcp with specific port
$ sudo lsof -itcp:8080

# List udp
$ sudo lsof -iudp
# Without port no convertion to name 
$ sudo lsof -iudp -P
# List udp with specific port
$ sudo lsof -iudp:54997
```
