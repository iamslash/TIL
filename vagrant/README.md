# Abstract

Local Machine 에서 VirtualBox 를 이용하여 provision 할 수 있는 application 이다.

# References

* [Vagrant Documentation @ vagrant](https://www.vagrantup.com/docs)

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

## Tutorial

```console
$ mkdir ~/my/vagrant/a
$ cd ~/my/vagrant/a
$ vagrant init
$ vim Vagrantfile
```

```Vagrantfile
# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|          # Vagrant Version
  config.vm.define:"Vagrant-VM01" do |cfg|  # Vagrant Virtual Machine
    cfg.vm.box = "centos/7"                 # Image name to download from Vagrant Cloud
    cfg.vm.provider:virtualbox do |vb|      # Vagrant Provider
      vb.name = "Vagrant-VM01"
    end
    cfg.vm.host_name = "Vagrant-VM01"       # hostname of CentOS
    cfg.vm.synced_folder ".", "/vagrant", disabled: true # Don't use synced_folder
    cfg.vm.network "public_network"         # Choose VirtualBox's NAT Interface, If there is no IP, this will use DHCP
    cfg.vm.network "forwarded_port", guest: 22, host: 19201, auto_correct: false, id: "ssh" # Forward port from 19201 of host to 22 of guest
  end
end
```

```console
$ vagrant up
$ vagrant ssh
$ vagrant destroy
```

# Advanced

## Vagrant with k8s

* [Kubernetes Setup Using Ansible and Vagrant](https://kubernetes.io/blog/2019/03/15/kubernetes-setup-using-ansible-and-vagrant/)

