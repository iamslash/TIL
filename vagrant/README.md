# Abstract

Local Machine 에서 VirtualBox 를 이용하여 provision 할 수 있는 application 이다.

# Materials

* [Vagrant 설치 및 기초사용방법](https://ossian.tistory.com/86)
  * [Vagrantfile 기초](https://ossian.tistory.com/87?category=771731)

# Install

## Install on windows

* [Install virtual box](https://www.virtualbox.org/wiki/Downloads)
* [Install Vagrant](https://www.vagrantup.com/downloads.html)

# Basic

## Commands

```bash
# Create Vagranfile
$ vagrant init

# Provision with Vagrantfile
$ vagrant up

# Stop virtual machine
$ vagrant halt

# Delete virtual machine
$ vagrant destory

# Connect to virtual machine with ssh
$ vagrant ssh

# Update virtual machine with Vagrantfile
$ vagrant provision
```

## Vagrantfile

```vagrant

```