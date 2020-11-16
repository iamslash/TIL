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

```Vagrantfile
IMAGE_NAME = "bento/ubuntu-16.04"
N = 2

Vagrant.configure("2") do |config|
    config.ssh.insert_key = false

    config.vm.provider "virtualbox" do |v|
        v.memory = 1024
        v.cpus = 2
    end
      
    config.vm.define "k8s-master" do |master|
        master.vm.box = IMAGE_NAME
        master.vm.network "private_network", ip: "192.168.50.10"
        master.vm.hostname = "k8s-master"
        master.vm.provision "ansible" do |ansible|
            ansible.playbook = "kubernetes-setup/master-playbook.yml"
            ansible.extra_vars = {
                node_ip: "192.168.50.10",
            }
        end
    end

    (1..N).each do |i|
        config.vm.define "node-#{i}" do |node|
            node.vm.box = IMAGE_NAME
            node.vm.network "private_network", ip: "192.168.50.#{i + 10}"
            node.vm.hostname = "node-#{i}"
            node.vm.provision "ansible" do |ansible|
                ansible.playbook = "kubernetes-setup/node-playbook.yml"
                ansible.extra_vars = {
                    node_ip: "192.168.50.#{i + 10}",
                }
            end
        end
    end
```

# Advanced

## Vagrant with k8s

* [Kubernetes Setup Using Ansible and Vagrant](https://kubernetes.io/blog/2019/03/15/kubernetes-setup-using-ansible-and-vagrant/)

