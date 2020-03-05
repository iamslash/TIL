# Abstract

크기가 매우 작은 리눅스 배포판이다. 

# Materials

* [알파인 리눅스(Alpine Linux)](https://www.lesstif.com/pages/viewpage.action?pageId=35356819)

# Basic

c runtime 으로 glibc 대신 [musl libc](https://en.wikipedia.org/wiki/Musl) 를 사용한다. shell commands 는 GNU util 대신 [busybox](https://en.wikipedia.org/wiki/BusyBox) 를 사용한다.

package manager 는 [apk](https://wiki.alpinelinux.org/wiki/Alpine_Linux_package_management) 를 사용한다.

# APK

```bash
# Refresh contents of packages
$ apk update

# Install vim
$ apk add vim

# Uninstall vim
$ apk del vim

# Search zsh
$ apk search zsh
$ apk info zsh

# List installed packages
$ apk info

# Show installed files of vim
$ apk info -L vim

# Apk uprade
$ apk update
$ apk upgrade
```